"""
Tests for the pure-numpy half of Phase 2.

fusion.py and speech.group_segments hold the logic that is easiest to get
subtly wrong and hardest to eyeball in real output -- a fusion bug does not
crash, it just quietly returns slightly worse results forever. Neither needs
torch, FAISS or a GPU, so these run in a fraction of a second:

    python3 test_fusion.py        (or: pytest test_fusion.py)
"""

import numpy as np

import fusion
from speech import group_segments


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


# --------------------------------------------------------------------------- #
# Normalising
# --------------------------------------------------------------------------- #

def test_zscore_centres_and_scales():
    z = fusion.zscore([1.0, 2.0, 3.0, 4.0])
    assert close(z.mean(), 0.0)
    assert close(z.std(), 1.0)
    assert z[0] < 0 < z[-1]


def test_zscore_survives_constant_input():
    # A one-frame video, or a query that scores identically everywhere.
    assert np.array_equal(fusion.zscore([0.3, 0.3, 0.3]), np.zeros(3))
    assert fusion.zscore([]).size == 0


def test_zscore_makes_different_scales_comparable():
    """The whole point: CLIP's narrow range must not be drowned out by MiniLM's."""
    clip_like = [0.20, 0.21, 0.34, 0.19]        # spread of ~0.15
    minilm_like = [0.05, 0.10, 0.80, 0.02]      # spread of ~0.78, same winner
    assert np.argmax(fusion.zscore(clip_like)) == np.argmax(fusion.zscore(minilm_like)) == 2
    # Raw, the transcript score at the winner is ~4x the visual one. Normalised,
    # they are within a hair of each other, so neither can bully the other.
    assert abs(fusion.zscore(clip_like)[2] - fusion.zscore(minilm_like)[2]) < 0.2


def test_zscore_favours_the_confident_modality():
    """A real spike should outvote a modality that found nothing in particular."""
    spike = fusion.zscore([0.1] * 20 + [0.9])       # one obvious answer
    flat = fusion.zscore(list(np.linspace(0.3, 0.4, 21)))   # no opinion
    assert spike.max() > flat.max()


def test_ranks_and_reciprocal_rank():
    assert list(fusion.ranks([0.1, 0.9, 0.5])) == [2, 0, 1]
    rr = fusion.reciprocal_rank([0.1, 0.9, 0.5])
    assert close(rr[1], 1 / fusion.RRF_K)
    assert rr[1] > rr[2] > rr[0]


# --------------------------------------------------------------------------- #
# Lining transcript chunks up with frames
# --------------------------------------------------------------------------- #

FRAMES = np.arange(0.0, 20.0)        # 20 frames, one per second


def test_alignment_covers_only_the_spoken_window():
    values, source = fusion.align_segments_to_frames(
        FRAMES, [5.0], [8.0], [2.5], pad=0.0)
    assert list(source[5:9]) == [0, 0, 0, 0]       # frames 5,6,7,8
    assert all(s == -1 for s in source[:5]) and all(s == -1 for s in source[9:])
    assert all(close(v, 2.5) for v in values[5:9])


def test_alignment_gives_silence_a_neutral_zero():
    values, source = fusion.align_segments_to_frames(
        FRAMES, [5.0], [8.0], [-3.0], pad=0.0)
    # Frames nobody spoke over must score 0 ("no opinion"), NOT the chunk's
    # value and not -inf -- otherwise silent-but-visually-perfect moments would
    # be punished for the silence.
    assert all(close(v, 0.0) for v in values[:5])
    assert all(close(v, -3.0) for v in values[5:9])


def test_alignment_keeps_the_best_of_overlapping_chunks():
    # Chunks overlap by design (see group_segments), so frames in the seam are
    # covered twice and should take the better score.
    values, source = fusion.align_segments_to_frames(
        FRAMES, [0.0, 4.0], [6.0, 10.0], [1.0, 9.0], pad=0.0)
    assert all(close(v, 9.0) for v in values[4:7])     # seam -> the 9.0 chunk
    assert all(s == 1 for s in source[4:7])
    assert all(close(v, 1.0) for v in values[0:4])     # before the seam


def test_alignment_pad_widens_coverage():
    bare, _ = fusion.align_segments_to_frames(FRAMES, [5.0], [6.0], [1.0], pad=0.0)
    padded, _ = fusion.align_segments_to_frames(FRAMES, [5.0], [6.0], [1.0], pad=2.0)
    assert (padded != 0).sum() > (bare != 0).sum()
    assert close(padded[3], 1.0) and close(padded[8], 1.0)


def test_alignment_ignores_a_chunk_between_two_frames():
    # Sampling every 5s, a 1-second chunk can fall in a gap. It must not crash
    # and must not be credited to a frame it does not cover.
    sparse = np.array([0.0, 5.0, 10.0])
    values, source = fusion.align_segments_to_frames(
        sparse, [6.0], [7.0], [5.0], pad=0.0)
    assert list(source) == [-1, -1, -1]
    assert np.array_equal(values, np.zeros(3))


def test_alignment_rejects_unsorted_frames():
    try:
        fusion.align_segments_to_frames([5.0, 1.0], [0.0], [9.0], [1.0])
    except ValueError:
        return
    raise AssertionError("unsorted frame_times should raise")


# --------------------------------------------------------------------------- #
# Fusing
# --------------------------------------------------------------------------- #

def test_agreement_beats_either_signal_alone():
    """
    The reason Phase 2 exists.

    Frame 10 looks somewhat right AND is spoken over by the right words.
    Frame 3 looks great but is talked over by nothing relevant; frame 17 has the
    right words over a nondescript picture. The moment where both agree wins.
    """
    visual = np.full(20, 0.20)
    visual[3] = 0.32                 # best-looking frame
    visual[10] = 0.26                # decent-looking frame
    seg_scores = np.array([0.05, 0.70, 0.62])
    seg_starts = np.array([0.0, 9.0, 16.0])
    seg_ends = np.array([8.0, 12.0, 19.0])

    out = fusion.fuse(FRAMES, visual, seg_scores, seg_starts, seg_ends,
                      mode="zscore", alpha=0.5, pad=0.0)
    assert int(np.argmax(out.score)) == 10
    # ...and neither single modality would have found it on its own.
    assert int(np.argmax(out.visual)) == 3
    assert int(np.argmax(out.speech)) in (9, 10, 11, 12)


def test_alpha_endpoints_isolate_each_modality():
    visual = np.full(20, 0.2)
    visual[3] = 0.9
    seg_scores, seg_starts, seg_ends = [0.1, 0.9], [0.0, 15.0], [5.0, 19.0]

    kw = dict(seg_scores=seg_scores, seg_starts=seg_starts, seg_ends=seg_ends, pad=0.0)
    visual_only = fusion.fuse(FRAMES, visual, alpha=1.0, **kw)
    speech_only = fusion.fuse(FRAMES, visual, alpha=0.0, **kw)
    assert int(np.argmax(visual_only.score)) == 3
    # Frames 15..19 tie at the top of the speech-only ranking (one chunk covers
    # them all), and argmax reports the first of a tie.
    assert int(np.argmax(speech_only.score)) == 15
    # alpha=1.0 must reproduce Phase 1 exactly, not approximately.
    assert np.array_equal(visual_only.score, fusion.zscore(visual))


def test_no_transcript_degrades_to_phase_one():
    visual = np.linspace(0.1, 0.5, 20)
    for empty in (None, [], np.array([])):
        out = fusion.fuse(FRAMES, visual, empty, alpha=0.5)
        # alpha is deliberately ignored here: half of nothing is still nothing,
        # and scaling the only real signal by 0.5 would change nothing but the
        # printed numbers.
        assert np.array_equal(out.score, fusion.zscore(visual))
        assert np.array_equal(out.speech, np.zeros(20))
        assert set(out.segment) == {-1}


def test_rrf_ignores_score_magnitude():
    starts, ends = [0.0], [19.0]
    modest = fusion.fuse(FRAMES, np.linspace(0.1, 0.2, 20), [0.3], starts, ends, mode="rrf")
    dramatic = fusion.fuse(FRAMES, np.linspace(1.0, 99.0, 20), [0.9], starts, ends, mode="rrf")
    # Same ordering in, same fused scores out -- only ranks survive.
    assert np.allclose(modest.score, dramatic.score)


def test_fuse_rejects_unknown_mode():
    try:
        fusion.fuse(FRAMES, np.zeros(20), mode="magic")
    except ValueError:
        return
    raise AssertionError("unknown mode should raise")


# --------------------------------------------------------------------------- #
# Picking results
# --------------------------------------------------------------------------- #

def test_top_moments_spreads_results_out():
    times = np.arange(0.0, 60.0)
    scores = np.zeros(60)
    scores[30:36] = [0.9, 0.89, 0.88, 0.87, 0.86, 0.85]   # one event, six frames
    scores[50] = 0.5                                      # a second, real event

    assert fusion.top_moments(times, scores, top=2, gap=10.0) == [30, 50]
    # With no gap enforced you get the same answer five times over.
    assert fusion.top_moments(times, scores, top=2, gap=0.0) == [30, 31]


def test_top_moments_handles_a_short_video():
    assert fusion.top_moments([0.0], [1.0], top=5) == [0]
    assert fusion.top_moments([], [], top=5) == []


# --------------------------------------------------------------------------- #
# Chunking the transcript
# --------------------------------------------------------------------------- #

def seg(start, end, text="word"):
    return {"start": start, "end": end, "text": text}


def test_group_segments_merges_short_segments():
    segments = [seg(i * 3.0, i * 3.0 + 2.5) for i in range(10)]   # 10 x ~3s
    chunks = group_segments(segments, chunk_seconds=10.0, overlap_seconds=0.0)
    assert len(chunks) < len(segments)
    assert all(c["end"] - c["start"] <= 10.0 + 3.0 for c in chunks)
    assert all(c["text"] for c in chunks)


def test_group_segments_covers_the_whole_transcript():
    segments = [seg(i * 3.0, i * 3.0 + 2.5, f"s{i}") for i in range(10)]
    chunks = group_segments(segments, chunk_seconds=10.0, overlap_seconds=2.0)
    assert close(chunks[0]["start"], segments[0]["start"])
    assert close(chunks[-1]["end"], segments[-1]["end"])
    # Every original segment's words must appear somewhere, or we have silently
    # dropped part of the lecture from the index.
    joined = " ".join(c["text"] for c in chunks)
    for i in range(10):
        assert f"s{i}" in joined


def test_group_segments_overlaps_so_sentences_stay_whole():
    segments = [seg(i * 3.0, i * 3.0 + 2.5, f"s{i}") for i in range(12)]
    overlapped = group_segments(segments, chunk_seconds=9.0, overlap_seconds=4.0)
    clean_cut = group_segments(segments, chunk_seconds=9.0, overlap_seconds=0.0)
    assert len(overlapped) > len(clean_cut)
    assert any(overlapped[i + 1]["start"] < overlapped[i]["end"]
               for i in range(len(overlapped) - 1))


def test_group_segments_always_terminates():
    segments = [seg(i * 3.0, i * 3.0 + 2.5) for i in range(20)]
    # Absurd settings (overlap >= chunk length) must still make progress.
    chunks = group_segments(segments, chunk_seconds=5.0, overlap_seconds=99.0)
    assert 0 < len(chunks) <= len(segments)


def test_group_segments_respects_the_character_cap():
    segments = [seg(i * 0.5, i * 0.5 + 0.4, "x" * 100) for i in range(20)]
    chunks = group_segments(segments, chunk_seconds=600.0, overlap_seconds=0.0, max_chars=250)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 400 for c in chunks)


def test_group_segments_handles_empty_and_single():
    assert group_segments([]) == []
    one = group_segments([seg(1.0, 2.0, "hello")])
    assert len(one) == 1 and one[0]["text"] == "hello"


def test_chunks_feed_alignment_without_gaps_during_speech():
    """End-to-end on the pure-logic path: chunk, then align, then every
    spoken frame must get a transcript score."""
    segments = [seg(i * 2.0, i * 2.0 + 1.8, f"s{i}") for i in range(10)]   # 0..19.8s
    chunks = group_segments(segments, chunk_seconds=6.0, overlap_seconds=2.0)
    _, source = fusion.align_segments_to_frames(
        FRAMES,
        [c["start"] for c in chunks],
        [c["end"] for c in chunks],
        np.arange(len(chunks), dtype="float64"),
        pad=0.0)
    assert (source >= 0).all(), "a frame during continuous speech was left uncovered"


if __name__ == "__main__":
    tests = sorted(k for k in dict(globals()) if k.startswith("test_"))
    for name in tests:
        globals()[name]()
        print(f"  ok  {name}")
    print(f"\n{len(tests)} tests passed")
