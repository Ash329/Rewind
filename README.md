# Rewind

Natural-language video search. Describe a moment and get the timestamp back.

Rewind indexes two things about a video and searches them together: what is
**on screen** (CLIP frame embeddings) and what is **being said** (Whisper
transcript embeddings). Searching both at once is the point — on a lecture, the
answer is usually the moment where the slide *and* the sentence agree.

```
$ python3 search.py index_lecture "why we use cross-entropy instead of MSE"
Loaded 3601 frames + 412 transcript chunks (en, Whisper base) from lecture.mp4

Top 5 for: "why we use cross-entropy instead of MSE"   [zscore, alpha=0.50]
  1.    27:14   score +3.880   (visual +1.790 | speech +5.970)
      "squared error punishes a confident wrong answer far too gently, which is why for classification we almost always reach for cross-entropy instead, and..."
  2.    31:02   score +1.640   (visual +2.310 | speech +0.970)
      "so this term here is the loss, and the gradient we actually want is the derivative of it with respect to these weights"
```

## Status

| Phase | What it adds | State |
|---|---|---|
| 1 | CLIP frame embeddings + FAISS, text→frame search | done |
| 2 | Whisper transcript, sentence embeddings, score fusion | **done** |
| 3 | OCR of on-screen text, hybrid keyword retrieval | not started |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No `ffmpeg` binary needed — audio is decoded through the ffmpeg libraries
bundled in faster-whisper's wheels. Expect ~4 GB of wheels plus ~1 GB of model
weights on first run (CLIP ~600 MB, Whisper `base` ~150 MB, MiniLM ~90 MB).

A GPU is optional. Whisper tries CUDA first and falls back to CPU on its own if
cuBLAS/cuDNN aren't where it wants them, which is common.

## Use

```bash
# index a video: frames + transcript
python3 index_video.py lecture.mp4

# search it
python3 search.py index_lecture "the slide about eigenvalues"
python3 search.py index_lecture                 # interactive, one query per line
```

Indexing writes a folder (`index_<video name>/` by default):

| File | What it is |
|---|---|
| `frames.faiss` | one 512-d CLIP vector per sampled frame |
| `timestamps.npy` | the time of each of those frames |
| `speech.faiss` | one 384-d MiniLM vector per transcript chunk |
| `segments.json` | the chunks those vectors came from (start, end, text) |
| `transcript.json` | Whisper's raw output — readable on its own, and a cache |
| `meta.json` | which models built this, and how |

### Re-indexing is cheap on purpose

Transcribing is the slow step (minutes), so `transcript.json` is kept and
reused. Everything downstream of it is seconds, and always rebuilt:

```bash
python3 index_video.py lecture.mp4 --every 0.5      # denser frames, same transcript
python3 index_video.py lecture.mp4 --speech-only \
        --chunk-seconds 30                          # re-chunk, no Whisper re-run
python3 index_video.py lecture.mp4 --retranscribe \
        --whisper-model small                       # redo it properly
python3 index_video.py lecture.mp4 --no-speech      # visual only
```

`--whisper-model` defaults to whatever built the existing transcript, so a plain
re-run never silently costs you twenty minutes. `tiny < base < small < medium <
large-v3`: each step is ~3x slower and noticeably better at technical vocabulary.
`small` is the sweet spot for real lectures.

## How the two scores get combined

The hard part of Phase 2 isn't Whisper, it's that the two similarity scores
aren't comparable. CLIP image-vs-text cosines mostly land in 0.15–0.35; MiniLM
text-vs-text cosines spread over −0.1–0.8. Adding them lets the transcript
outvote the picture every time regardless of which one is right.

So each score is first rewritten as **how unusual it is for this query** (a
z-score), and only then averaged. That has a useful side effect: a model that
found a real match sticks out far above its own average (+6σ), while a model
that found nothing has a flat spread where even its best guess is +2σ — so the
modality that is actually confident wins the vote, with no per-query tuning.

Frames are instants and transcript chunks are spans, so results are reported on
the frame timeline: each frame takes the best score among the chunks being
spoken over it, and frames nobody spoke over score 0 — "no opinion" — rather
than being punished for the silence.

```bash
python3 search.py idx "chain rule" --fuse visual    # Phase 1 behaviour
python3 search.py idx "chain rule" --fuse speech    # transcript only
python3 search.py idx "chain rule" --alpha 0.7      # lean visual (1.0 … 0.0)
python3 search.py idx "chain rule" --fuse rrf       # combine ranks, not scores
```

Running one query three ways is how you tell whether speech actually helped.
`--fuse rrf` (reciprocal rank fusion) throws away the score magnitudes and
merges the two rankings instead; it needs no calibration, but it can't tell a
runaway best match from a barely-ahead one, which is why `zscore` is the default.

Transcript chunks are ~18 s of speech with ~4 s of overlap (`--chunk-seconds`,
`--chunk-overlap`). Whisper's own segments are often a single clause — "Okay, so."
is useless to embed, because it sits equally near every query. Overlapping means
a sentence split across a boundary still lands whole inside one chunk.

An index built by Phase 1 (or with `--no-speech`) still works: search notices
there is no transcript and falls back to visual-only ranking.

## Tests

```bash
python3 test_fusion.py      # or: pytest test_fusion.py
```

25 tests over `fusion.py` and `speech.group_segments` — the normalising, the
chunk→frame alignment, the fusion itself, and the chunking. They need neither
torch nor a GPU, which is the reason that logic lives in files with no heavy
imports: a fusion bug doesn't crash, it just quietly returns slightly worse
results forever.
