"""
Turning two different kinds of similarity score into one ranked list.

Phase 1 had one number per moment: how much does this *frame* look like the
query. Phase 2 adds a second: how much does what was *said* around this moment
mean the same thing as the query.

The catch is that the two numbers are not comparable. They come from different
models trained in different ways, so they live on different scales:

    CLIP image-vs-text cosine similarity   usually lands in  0.15 .. 0.35
    MiniLM text-vs-text cosine similarity  usually lands in -0.10 .. 0.80

Adding those together would quietly let the transcript outvote the picture
every single time, no matter which one was actually right. So before mixing
them we convert each score into "how unusual is this, for this query?" -- and
only then take a weighted average.

Everything in this file is plain numpy on purpose: no torch, no FAISS. That
keeps the part that is easiest to get subtly wrong easy to read and easy to
test (see test_fusion.py).
"""

from typing import NamedTuple

import numpy as np

# How much weight the visual side gets. 1.0 = frames only (Phase 1 behaviour),
# 0.0 = transcript only, 0.5 = treat both as equally trustworthy.
DEFAULT_ALPHA = 0.5

# Seconds of slack when deciding which transcript chunk covers a frame. A
# lecturer usually says "as you can see on this slide" a beat before or after
# the slide itself is up, so a little padding helps rather than hurts.
DEFAULT_PAD = 2.0

# The constant in Reciprocal Rank Fusion. 60 is the value from the original
# paper and nobody has found much reason to change it. Bigger = flatter, i.e.
# the gap between rank 1 and rank 2 matters less.
RRF_K = 60


class Fused(NamedTuple):
    """One entry per sampled frame."""
    score: np.ndarray      # the combined score we actually rank by
    visual: np.ndarray     # the visual half, after normalising
    speech: np.ndarray     # the transcript half, after normalising
    segment: np.ndarray    # which transcript chunk fed this frame (-1 = silence)


def zscore(values):
    """
    Rewrite each score as "how many standard deviations above average is it?".

    This is the trick that makes two models comparable. It also has a property
    that is exactly what we want here: a model that found a *real* match sticks
    out a long way above its own average (say +6), while a model that found
    nothing much has a flat spread where even its best guess is only +2. So the
    modality that is actually confident naturally wins the vote, without us
    having to hand-tune anything per query.
    """
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return values
    spread = values.std()
    if spread < 1e-8:
        # Every score identical (e.g. a one-frame video). No information here,
        # so say so with zeros instead of dividing by ~0 and getting garbage.
        return np.zeros_like(values)
    return (values - values.mean()) / spread


def ranks(values):
    """Position of each score in the sorted-best-first order. 0 = best."""
    values = np.asarray(values, dtype="float64")
    order = np.argsort(-values, kind="stable")
    out = np.empty(values.size, dtype="int64")
    out[order] = np.arange(values.size)
    return out


def reciprocal_rank(values):
    """
    Score each item by 1 / (K + its rank) instead of by its raw similarity.

    This throws away *how far apart* the scores were and keeps only the order,
    which makes it immune to the scale problem described at the top of the file.
    Cheaper insurance than z-scoring, but blunter: a runaway best match and a
    barely-ahead best match both just get "rank 0".
    """
    return 1.0 / (RRF_K + ranks(values))


def align_segments_to_frames(frame_times, seg_starts, seg_ends, seg_values, pad=DEFAULT_PAD):
    """
    Spread per-chunk transcript scores out onto the per-frame timeline.

    The two halves of the index disagree about what a "result" is: frames are
    instants (one per second) and transcript chunks are spans (~18 seconds of
    speech). We settle on frames, because a frame is something the user can
    actually jump to and look at. So for each frame we ask: of the chunks being
    spoken at that moment, which scored best?

    `seg_values` must already be normalised (z-scores or reciprocal ranks),
    because frames that nobody was talking over are given 0.0 -- "no opinion" --
    and that only means the right thing on a normalised scale.

    Returns (values, source); source[i] is the chunk that won frame i, or -1 if
    the room was silent there.
    """
    frame_times = np.asarray(frame_times, dtype="float64")
    if np.any(np.diff(frame_times) < 0):
        raise ValueError("frame_times must be sorted ascending")

    values = np.full(frame_times.size, -np.inf, dtype="float64")
    source = np.full(frame_times.size, -1, dtype="int64")

    seg_starts = np.asarray(seg_starts, dtype="float64")
    seg_ends = np.asarray(seg_ends, dtype="float64")
    seg_values = np.asarray(seg_values, dtype="float64")

    for c in range(seg_values.size):
        # Frames inside this chunk's spoken window are the contiguous slice
        # [lo, hi) of the timeline, because frame_times is sorted.
        lo = int(np.searchsorted(frame_times, seg_starts[c] - pad, side="left"))
        hi = int(np.searchsorted(frame_times, seg_ends[c] + pad, side="right"))
        if hi <= lo:
            continue                     # chunk falls between two sampled frames
        window = slice(lo, hi)
        better = seg_values[c] > values[window]
        values[window] = np.where(better, seg_values[c], values[window])
        source[window] = np.where(better, c, source[window])

    values[source < 0] = 0.0             # silence: let the visuals decide alone
    return values, source


def fuse(frame_times, visual_scores, seg_scores=None, seg_starts=None, seg_ends=None,
         mode="zscore", alpha=DEFAULT_ALPHA, pad=DEFAULT_PAD):
    """
    Combine the visual and transcript scores into one number per frame.

    mode="zscore"  normalise both sides by spread, then weighted-average them.
    mode="rrf"     ignore the raw scores, combine the two rankings instead.

    With no transcript (a Phase 1 index, or --no-speech) this degrades to plain
    visual search: the normalised visual score is returned as-is and `alpha` is
    ignored, so the ranking is identical to Phase 1's.
    """
    if mode not in ("zscore", "rrf"):
        raise ValueError(f"unknown fusion mode: {mode!r}")

    normalise = zscore if mode == "zscore" else reciprocal_rank
    visual = normalise(visual_scores)
    n = visual.size

    have_speech = seg_scores is not None and len(seg_scores) > 0
    if not have_speech:
        return Fused(score=visual.copy(), visual=visual,
                     speech=np.zeros(n), segment=np.full(n, -1, dtype="int64"))

    speech, segment = align_segments_to_frames(
        frame_times, seg_starts, seg_ends, normalise(seg_scores), pad=pad)

    score = alpha * visual + (1.0 - alpha) * speech
    return Fused(score=score, visual=visual, speech=speech, segment=segment)


def top_moments(frame_times, scores, top=5, gap=10.0):
    """
    Best `top` frames, but never two within `gap` seconds of each other.

    Same reason as Phase 1: consecutive seconds of a lecture are nearly the same
    picture and nearly the same sentence, so an un-deduplicated top 5 would be
    12:31, 12:32, 12:33, 12:34, 12:35 -- five copies of one answer.
    """
    frame_times = np.asarray(frame_times, dtype="float64")
    kept = []
    for i in np.argsort(-np.asarray(scores, dtype="float64"), kind="stable"):
        t = float(frame_times[i])
        if all(abs(t - float(frame_times[j])) >= gap for j in kept):
            kept.append(int(i))
            if len(kept) == top:
                break
    return kept
