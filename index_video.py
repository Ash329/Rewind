import argparse
import json
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import speech
from device import free_gpu, pick_device

# A small, fast CLIP model. Every vector it produces has 512 numbers.
MODEL_NAME = "openai/clip-vit-base-patch32"


def sample_frames(video_path, every_seconds=1.0):
    """
    Walk through the video and yield (timestamp_in_seconds, image)
    once every `every_seconds`.

    A video is just a stack of still images (frames) played quickly,
    e.g. 30 frames per second (fps). To get 1 frame per second from a
    30 fps video we keep frame 0, 30, 60, 90, ...
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError("Could not read the video's frame rate.")
    step = max(1, round(fps * every_seconds))   # keep every `step`-th frame

    frame_idx = 0
    while True:
        # grab() moves to the next frame cheaply; retrieve() actually decodes it.
        # We only decode the frames we keep, which saves time.
        if not cap.grab():
            break                                # end of video
        if frame_idx % step == 0:
            ok, frame_bgr = cap.retrieve()
            if ok:
                # OpenCV stores colours as BGR; everything else expects RGB.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_idx / fps, Image.fromarray(frame_rgb)
        frame_idx += 1

    cap.release()


@torch.no_grad()   # we're only using the model, not training it -> no gradients needed
def embed_images(images, model, processor, device):
    """Turn a list of images into a (N, 512) array of unit-length vectors."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    # Normalise each vector to length 1. After this, the dot product of two
    # vectors equals their cosine similarity (how much they point the same way).
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")   # FAISS wants float32


def build_visual_index(video_path, out_dir, every_seconds, batch_size):
    """Phase 1: one CLIP vector per sampled frame -> frames.faiss + timestamps.npy"""
    device = pick_device()
    print(f"Loading {MODEL_NAME} on {device} (first run downloads ~600 MB)...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    all_vectors, all_timestamps = [], []
    batch_imgs, batch_ts = [], []
    start = time.time()

    def flush():
        """Embed whatever is waiting in the current batch."""
        if batch_imgs:
            all_vectors.append(embed_images(batch_imgs, model, processor, device))
            all_timestamps.extend(batch_ts)
            batch_imgs.clear()
            batch_ts.clear()

    for ts, img in sample_frames(video_path, every_seconds):
        batch_imgs.append(img)
        batch_ts.append(ts)
        if len(batch_imgs) == batch_size:
            flush()
            print(f"  embedded up to {ts/60:6.1f} min  ({len(all_timestamps)} frames)", end="\r")
    flush()
    print()

    if not all_vectors:
        raise RuntimeError("No frames were read from the video.")

    vectors = np.vstack(all_vectors)                  # shape: (num_frames, 512)
    timestamps = np.array(all_timestamps, dtype="float32")

    # IndexFlatIP = "flat" (compare against every vector, exact, no tricks)
    #               + "IP" (score = inner/dot product = cosine sim, since we normalised).
    # For one lecture (~3,600 vectors) this is instant. Fancier index types only
    # matter once you have millions of vectors.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(out_dir / "frames.faiss"))
    np.save(out_dir / "timestamps.npy", timestamps)

    # Whisper is about to want the whole GPU to itself.
    del model, processor
    free_gpu()

    print(f"  {index.ntotal} frames in {time.time() - start:.0f}s")
    return {"visual_model": MODEL_NAME,
            "every_seconds": every_seconds,
            "num_frames": int(index.ntotal)}


def build_speech_index(video_path, out_dir, args):
    """
    Phase 2: Whisper transcript -> overlapping chunks -> speech.faiss + segments.json

    Transcribing is the slow part (minutes), while chunking and embedding are the
    fast part (seconds). So they are cached separately: transcript.json survives
    between runs, and the chunk/embed step always re-runs with whatever
    --chunk-seconds / --text-model you asked for this time.
    """
    start = time.time()
    cached = speech.read_transcript(out_dir)
    reusable = (
        cached is not None
        and not args.retranscribe
        and cached.get("video") == str(Path(video_path).resolve())
        # Only an *explicitly requested* different Whisper model forces a redo;
        # the flag defaults to None so a plain re-run never costs you 20 minutes.
        and (args.whisper_model is None or args.whisper_model == cached.get("whisper_model"))
    )

    if reusable:
        segments, info = cached["segments"], {k: v for k, v in cached.items()
                                             if k not in ("segments", "video")}
        print(f"Reusing transcript.json: {len(segments)} segments from Whisper "
              f"'{info.get('whisper_model')}' (--retranscribe to redo it)")
    else:
        segments, info = speech.transcribe(
            video_path,
            model_name=args.whisper_model or speech.DEFAULT_WHISPER_MODEL,
            language=args.language,
            device=args.whisper_device,
            vad=not args.no_vad,
        )
        speech.write_transcript(out_dir, video_path, segments, info)

    if not segments:
        print("  no speech found -- indexing visuals only.")
        for stale in ("speech.faiss", "segments.json"):
            (out_dir / stale).unlink(missing_ok=True)
        return None

    chunks = speech.group_segments(segments,
                                   chunk_seconds=args.chunk_seconds,
                                   overlap_seconds=args.chunk_overlap)
    print(f"  {len(segments)} segments -> {len(chunks)} chunks of ~{args.chunk_seconds:.0f}s")

    print(f"Embedding transcript with {args.text_model}...")
    encoder = speech.load_text_encoder(args.text_model)
    vectors = speech.embed_texts(encoder, [c["text"] for c in chunks])
    del encoder
    free_gpu()

    index = faiss.IndexFlatIP(vectors.shape[1])       # 384 dims for MiniLM
    index.add(vectors)
    faiss.write_index(index, str(out_dir / "speech.faiss"))
    (out_dir / "segments.json").write_text(json.dumps(chunks, indent=2))

    print(f"  {index.ntotal} chunks embedded in {time.time() - start:.0f}s")
    return {"whisper_model": info.get("whisper_model"),
            "text_model": args.text_model,
            "language": info.get("language"),
            "duration": info.get("duration"),
            "chunk_seconds": args.chunk_seconds,
            "chunk_overlap": args.chunk_overlap,
            "num_segments": len(segments),
            "num_chunks": int(index.ntotal)}


def main():
    parser = argparse.ArgumentParser(
        description="Index a video: CLIP frame vectors + Whisper transcript vectors.")
    parser.add_argument("video", help="Path to the video file (mp4, mkv, ...)")
    parser.add_argument("--out", default=None, help="Output folder (default: index_<video name>)")
    parser.add_argument("--every", type=float, default=1.0, help="Seconds between sampled frames")
    parser.add_argument("--batch", type=int, default=32, help="Frames sent to CLIP at once")

    speech_opts = parser.add_argument_group("speech (phase 2)")
    speech_opts.add_argument("--no-speech", action="store_true",
                             help="Skip transcription; keep any transcript already indexed")
    speech_opts.add_argument("--speech-only", action="store_true",
                             help="Only (re)build the speech half of an existing index")
    speech_opts.add_argument("--whisper-model", default=None,
                             help=f"tiny|base|small|medium|large-v3 "
                                  f"(default: {speech.DEFAULT_WHISPER_MODEL}, or whatever "
                                  f"transcript.json was built with)")
    speech_opts.add_argument("--whisper-device", default=None,
                             help="Force 'cpu' or 'cuda' for Whisper (default: try cuda, fall back)")
    speech_opts.add_argument("--language", default=None,
                             help="Spoken language, e.g. 'en' (default: let Whisper detect it)")
    speech_opts.add_argument("--retranscribe", action="store_true",
                             help="Re-run Whisper even if transcript.json already exists")
    speech_opts.add_argument("--no-vad", action="store_true",
                             help="Don't skip silence (slower, and may hallucinate over noise)")
    speech_opts.add_argument("--text-model", default=speech.DEFAULT_TEXT_MODEL,
                             help="Sentence-embedding model for transcript chunks")
    speech_opts.add_argument("--chunk-seconds", type=float, default=speech.DEFAULT_CHUNK_SECONDS,
                             help="Rough length of each embedded transcript chunk")
    speech_opts.add_argument("--chunk-overlap", type=float, default=speech.DEFAULT_CHUNK_OVERLAP,
                             help="How far each chunk reaches back into the previous one")
    args = parser.parse_args()

    if args.no_speech and args.speech_only:
        parser.error("--no-speech and --speech-only are opposites; pick one.")

    video_path = Path(args.video)
    out_dir = Path(args.out or f"index_{video_path.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    overall_start = time.time()

    if args.speech_only:
        missing = [f for f in ("frames.faiss", "timestamps.npy") if not (out_dir / f).exists()]
        if missing:
            parser.error(f"--speech-only needs an existing index in {out_dir}/ "
                         f"(missing: {', '.join(missing)})")
        print(f"Keeping the {meta.get('num_frames', '?')} frames already in {out_dir}/")
    else:
        meta.update(build_visual_index(video_path, out_dir, args.every, args.batch))

    meta["video"] = str(video_path.resolve())

    if args.no_speech and meta.get("speech"):
        # The transcript depends only on the audio, so re-sampling frames at a
        # different --every does NOT invalidate it -- that is the point of this
        # flag. Pointing a *different video* at this folder does invalidate it.
        cached = speech.read_transcript(out_dir)
        if cached is not None and cached.get("video") == meta["video"]:
            print(f"Keeping the existing transcript "
                  f"({meta['speech'].get('num_chunks', '?')} chunks)")
        else:
            # Only forget it in meta.json -- search.py reads the speech half only
            # when meta says it is there, and nothing expensive gets deleted.
            print("Ignoring the transcript in this folder: it is from a different video.")
            meta.pop("speech")
    elif not args.no_speech:
        speech_meta = build_speech_index(video_path, out_dir, args)
        if speech_meta:
            meta["speech"] = speech_meta
        else:
            meta.pop("speech", None)

    meta_path.write_text(json.dumps(meta, indent=2))

    chunks = meta.get("speech", {}).get("num_chunks", 0)
    print(f"\nDone in {time.time() - overall_start:.0f}s -> {out_dir}/")
    print(f"  {meta.get('num_frames', 0)} frames, {chunks} transcript chunks")
    if not chunks:
        print("  (visual-only index -- search will fall back to Phase 1 behaviour)")


if __name__ == "__main__":
    main()
