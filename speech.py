"""
The speech half of the index: what was said, when, and what it means.

Three steps:

  1. Transcribe   Whisper turns the audio into timed segments, e.g.
                  {start: 743.2, end: 751.8, text: "so the gradient flows
                  backwards through each layer"}.

  2. Chunk        Whisper's own segments are often a single short clause
                  ("Okay."). We merge neighbours into ~18-second, slightly
                  overlapping chunks, because a sentence-embedding model needs a
                  bit of context to produce a meaningful vector.

  3. Embed        A sentence-embedding model turns each chunk into a vector, in
                  a space where "backprop" and "gradients flowing backwards"
                  land near each other. This is a *different* space from CLIP's
                  -- that is the whole reason fusion.py has to normalise before
                  mixing the two.

The heavy imports happen inside the functions, not at the top of the file, so
that the chunking logic stays importable (and testable) without faster-whisper
or sentence-transformers installed.
"""

import json
from pathlib import Path

# Whisper size. tiny < base < small < medium < large-v3: each step up is roughly
# 3x slower and noticeably better at proper nouns and technical vocabulary.
# "base" is the friendly default; "small" is the sweet spot for real lectures.
DEFAULT_WHISPER_MODEL = "base"

# 384 numbers per vector, fast on CPU, and still one of the best quality/size
# trade-offs available. Unlike the bge/e5 families it needs no "query:" prefix,
# which keeps the search side simple.
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_SECONDS = 18.0
DEFAULT_CHUNK_OVERLAP = 4.0
MAX_CHUNK_CHARS = 1000      # MiniLM truncates past ~256 word pieces anyway


# --------------------------------------------------------------------------- #
# 1. Transcribe
# --------------------------------------------------------------------------- #

def transcribe(video_path, model_name=DEFAULT_WHISPER_MODEL, language=None,
               device=None, vad=True, progress=True):
    """
    Run Whisper over the video's audio track.

    Returns (segments, info): a list of {"start", "end", "text"} dicts and a
    dict of what Whisper noticed (detected language, audio duration, ...).

    We use faster-whisper rather than openai-whisper for two reasons: it is
    several times quicker for the same model, and it decodes audio through
    bundled ffmpeg libraries, so there is no need for an `ffmpeg` binary on PATH.
    """
    from device import pick_device

    if device:
        candidates = [device]
    else:
        # faster-whisper on CUDA needs cuBLAS and cuDNN to be findable, which is
        # not always true even when torch is perfectly happy. So treat the GPU as
        # an optimistic first try and fall back rather than dying.
        candidates = ["cuda", "cpu"] if pick_device() == "cuda" else ["cpu"]

    for attempt, candidate in enumerate(candidates):
        try:
            return _transcribe_on(video_path, model_name, language, candidate, vad, progress)
        except Exception as error:
            if attempt == len(candidates) - 1:
                raise                       # nothing left to fall back to
            # Anything from "cuDNN not found" at load time to running out of VRAM
            # part-way through. Either way the CPU can still finish the job.
            print(f"  {candidate} transcription failed ({type(error).__name__}: {error})")
            print("  falling back to CPU -- slower, same result.")


def _transcribe_on(video_path, model_name, language, device, vad, progress):
    from faster_whisper import WhisperModel

    # int8 on CPU is ~3x faster than float32 for a barely measurable accuracy
    # cost; float16 is the natural choice on a GPU.
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper '{model_name}' on {device} ({compute_type})...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    # vad_filter skips stretches with no voice in them. On a lecture recording
    # that is a real speed-up, and it stops Whisper from inventing text out of
    # room noise during silences (its most notorious failure mode).
    segment_iter, info = model.transcribe(
        str(video_path),
        language=language,
        vad_filter=vad,
        beam_size=5,
    )
    print(f"  language: {info.language} (confidence {info.language_probability:.2f}), "
          f"audio: {info.duration / 60:.1f} min")

    segments = []
    # transcribe() is lazy: the work actually happens as we walk this generator,
    # which is what lets us show progress instead of staring at a blank screen.
    for segment in segment_iter:
        text = segment.text.strip()
        if text:
            segments.append({"start": float(segment.start),
                             "end": float(segment.end),
                             "text": text})
        if progress and segments:
            done = segment.end / 60
            total = max(info.duration, segment.end) / 60
            print(f"  transcribed {done:6.1f} / {total:.1f} min "
                  f"({len(segments)} segments)", end="\r")
    if progress:
        print()

    return segments, {
        "whisper_model": model_name,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "vad": bool(vad),
    }


# --------------------------------------------------------------------------- #
# 2. Chunk
# --------------------------------------------------------------------------- #

def group_segments(segments, chunk_seconds=DEFAULT_CHUNK_SECONDS,
                   overlap_seconds=DEFAULT_CHUNK_OVERLAP, max_chars=MAX_CHUNK_CHARS):
    """
    Merge Whisper's short segments into longer, overlapping chunks.

    Why bother: "Okay, so." is a useless thing to embed -- it is near-identical
    to every other filler phrase in the video, so it matches every query equally
    and tells you nothing. Giving the embedder ~18 seconds of speech gives it an
    actual idea to encode.

    Why overlap: a chunk boundary that lands in the middle of "...which is why we
    use | cross-entropy loss here" would split the one phrase the user is going
    to search for. Restarting each chunk `overlap_seconds` before the previous
    one ended means every sentence sits whole inside at least one chunk.
    """
    if not segments:
        return []
    overlap_seconds = min(overlap_seconds, chunk_seconds * 0.5)

    chunks = []
    i, n = 0, len(segments)
    while i < n:
        start = segments[i]["start"]
        end = segments[i]["end"]
        parts, chars = [], 0

        j = i
        while j < n:
            segment = segments[j]
            too_long = segment["end"] - start > chunk_seconds
            too_wordy = chars + len(segment["text"]) > max_chars
            if j > i and (too_long or too_wordy):
                break                     # j > i: always take at least one
            parts.append(segment["text"])
            chars += len(segment["text"])
            end = segment["end"]
            j += 1

        chunks.append({"start": float(start), "end": float(end),
                       "text": " ".join(parts).strip()})
        if j >= n:
            break

        # Next chunk begins at the first segment that reaches past the overlap
        # point. The `i + 1` floor guarantees we always move forward, so this
        # terminates no matter how the thresholds are set.
        k = i + 1
        while k < n and segments[k]["end"] <= end - overlap_seconds:
            k += 1
        i = k
    return chunks


# --------------------------------------------------------------------------- #
# 3. Embed
# --------------------------------------------------------------------------- #

def load_text_encoder(model_name=DEFAULT_TEXT_MODEL, device=None):
    """Load the sentence-embedding model. Used when indexing *and* when searching."""
    from sentence_transformers import SentenceTransformer

    from device import pick_device
    return SentenceTransformer(model_name, device=device or pick_device())


def embed_texts(encoder, texts, batch_size=64):
    """
    Turn text into a (N, 384) array of unit-length float32 vectors.

    normalize_embeddings=True is the same trick Phase 1 does by hand: once every
    vector has length 1, a dot product *is* the cosine similarity, so FAISS's
    inner-product index gives us cosine search for free.
    """
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")        # FAISS wants float32


# --------------------------------------------------------------------------- #
# The raw transcript on disk, so a 20-minute Whisper run happens only once
# --------------------------------------------------------------------------- #

TRANSCRIPT_FILE = "transcript.json"


def write_transcript(out_dir, video_path, segments, info):
    payload = dict(info)
    payload["video"] = str(Path(video_path).resolve())
    payload["segments"] = segments
    (Path(out_dir) / TRANSCRIPT_FILE).write_text(json.dumps(payload, indent=2))


def read_transcript(out_dir):
    """The cached transcript, or None if there isn't one."""
    path = Path(out_dir) / TRANSCRIPT_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload.get("segments"), list):
        return None
    return payload
