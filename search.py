import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

import fusion
import speech
from device import pick_device

# How much of a matching transcript chunk to print under each hit.
TEXT_WIDTH = 150


def format_ts(seconds):
    """3725.0 -> '1:02:05'"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@torch.no_grad()
def embed_text_clip(text, model, processor, device):
    """Turn a sentence into a unit-length 512-number vector in the SAME space as the images."""
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")   # shape (1, 512)


def score_everything(index, query_vector):
    """
    Similarity between the query and *every* vector in an index, in index order.

    Phase 1 asked FAISS for the top k and ranked those. Fusion needs more than
    that: to turn a score into "how unusual is this" we need the whole
    distribution, and to line transcript chunks up with frames we need a score
    for every frame, not just the shortlist. For one lecture that is a few
    thousand dot products -- genuinely free. (At a million vectors you would
    switch to retrieving a deep shortlist per modality and normalising within it.)

    FAISS returns results best-first, so we scatter them back into index order.
    """
    scores, ids = index.search(query_vector, index.ntotal)
    ids, scores = ids[0], scores[0]
    # A flat index hands back every vector exactly once, so this fills up
    # completely. The fill value is only insurance for index types that can
    # return fewer than k hits (and padding with the worst seen score keeps the
    # normalising in fusion.py well-behaved, which -inf would not).
    ordered = np.full(index.ntotal, float(scores.min()), dtype="float64")
    found = ids >= 0
    ordered[ids[found]] = scores[found]
    return ordered


class Index:
    """Everything on disk for one video, loaded and ready to query."""

    def __init__(self, index_dir):
        index_dir = Path(index_dir)
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.frames = faiss.read_index(str(index_dir / "frames.faiss"))
        self.timestamps = np.load(index_dir / "timestamps.npy").astype("float64")
        self.device = pick_device()

        # MUST be the same model that built the index, or the vectors live in
        # different "spaces" and the comparison is meaningless. ("model" is what
        # Phase 1 called this key, so old index folders still load.)
        visual_model = self.meta.get("visual_model") or self.meta["model"]
        self.clip = CLIPModel.from_pretrained(visual_model).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(visual_model)

        # The speech half is optional: a Phase 1 folder, a silent video, or
        # --no-speech all leave us with visuals only.
        self.speech = None
        speech_meta = self.meta.get("speech")
        if speech_meta and (index_dir / "speech.faiss").exists():
            segments = json.loads((index_dir / "segments.json").read_text())
            speech_index = faiss.read_index(str(index_dir / "speech.faiss"))
            if speech_index.ntotal != len(segments):
                raise RuntimeError(
                    f"{index_dir}/ is inconsistent: {speech_index.ntotal} speech vectors "
                    f"but {len(segments)} chunks in segments.json. Re-run index_video.py "
                    f"--speech-only to rebuild it.")
            self.speech = speech_index
            self.segments = segments
            self.seg_starts = np.array([s["start"] for s in segments], dtype="float64")
            self.seg_ends = np.array([s["end"] for s in segments], dtype="float64")
            self.text_encoder = speech.load_text_encoder(
                speech_meta.get("text_model", speech.DEFAULT_TEXT_MODEL), self.device)

    @property
    def has_speech(self):
        return self.speech is not None

    def describe(self):
        name = Path(self.meta["video"]).name
        parts = [f"{self.frames.ntotal} frames"]
        if self.has_speech:
            parts.append(f"{self.speech.ntotal} transcript chunks "
                         f"({self.meta['speech'].get('language', '?')}, "
                         f"Whisper {self.meta['speech'].get('whisper_model', '?')})")
        return f"Loaded {' + '.join(parts)} from {name}"

    def query(self, text, mode="zscore", alpha=fusion.DEFAULT_ALPHA, pad=fusion.DEFAULT_PAD):
        """Score every moment in the video against `text`."""
        visual = score_everything(
            self.frames, embed_text_clip(text, self.clip, self.clip_processor, self.device))

        if not self.has_speech:
            return fusion.fuse(self.timestamps, visual, mode=mode)

        # The same words, embedded a second time by a different model, because
        # CLIP's text tower and MiniLM live in unrelated vector spaces. CLIP's is
        # the one that understands pictures; MiniLM's is the one that understands
        # sentences. Each query vector is only ever compared inside its own space.
        query_vector = speech.embed_texts(self.text_encoder, [text])
        seg_scores = score_everything(self.speech, query_vector)

        return fusion.fuse(self.timestamps, visual, seg_scores,
                           self.seg_starts, self.seg_ends,
                           mode=mode, alpha=alpha, pad=pad)


def main():
    parser = argparse.ArgumentParser(
        description="Search a video index with text, fusing frames and transcript.")
    parser.add_argument("index_dir", help="Folder created by index_video.py")
    parser.add_argument("query", nargs="?", help="Text to search for (omit for interactive mode)")
    parser.add_argument("--top", type=int, default=5, help="How many results to print")
    parser.add_argument("--gap", type=float, default=10.0,
                        help="Min seconds between results (0 = allow near-duplicates)")
    parser.add_argument("--fuse", default="zscore",
                        choices=["zscore", "rrf", "visual", "speech"],
                        help="zscore: weighted average of normalised scores (default). "
                             "rrf: combine the two rankings instead of the scores. "
                             "visual/speech: one modality only, for comparison.")
    parser.add_argument("--alpha", type=float, default=fusion.DEFAULT_ALPHA,
                        help="Weight on the visual side: 1.0 = frames only, 0.0 = speech only")
    parser.add_argument("--pad", type=float, default=fusion.DEFAULT_PAD,
                        help="Seconds of slack when matching transcript chunks to frames")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0.0 and 1.0")

    # The two ablation modes are just fixed weights on the normal z-score fusion.
    # Being able to run the exact same query three ways is how you tell whether
    # adding speech actually helped.
    mode, alpha = args.fuse, args.alpha
    if args.fuse == "visual":
        mode, alpha = "zscore", 1.0
    elif args.fuse == "speech":
        mode, alpha = "zscore", 0.0

    index = Index(args.index_dir)
    print(index.describe())

    if not index.has_speech:
        if args.fuse == "speech":
            parser.error("this index has no transcript; run index_video.py --speech-only first")
        print("No transcript in this index -- searching frames only (Phase 1 behaviour).")
    elif alpha == 1.0:
        print("Visual only: the transcript is loaded but weighted 0.")
    elif alpha == 0.0:
        print("Speech only: frames are weighted 0.")

    def run(query):
        scored = index.query(query, mode=mode, alpha=alpha, pad=args.pad)
        label = f"{args.fuse}" + ("" if args.fuse in ("visual", "speech")
                                  else f", alpha={alpha:.2f}")
        print(f'\nTop {args.top} for: "{query}"   [{label}]')

        hits = fusion.top_moments(index.timestamps, scored.score, args.top, args.gap)
        if not hits:
            print("  (nothing to show)")
            return

        for rank, i in enumerate(hits, 1):
            ts = format_ts(index.timestamps[i])
            # RRF scores all sit just under 1/60, so they need the extra digits.
            places = 5 if mode == "rrf" else 3
            line = f"  {rank}. {ts:>8}   score {scored.score[i]:+.{places}f}"
            if index.has_speech:
                line += (f"   (visual {scored.visual[i]:+.{places}f} | "
                         f"speech {scored.speech[i]:+.{places}f})")
            print(line)

            # Printing what was actually said is the clearest sign that the
            # second modality is doing something: you can read why it matched.
            seg = int(scored.segment[i])
            if index.has_speech and seg >= 0:
                said = " ".join(index.segments[seg]["text"].split())
                if len(said) > TEXT_WIDTH:
                    said = said[:TEXT_WIDTH - 1].rstrip() + "..."
                print(f'      "{said}"')

    if args.query:
        run(args.query)
    else:
        try:
            while True:
                q = input("\nquery> ").strip()
                if q:
                    run(q)
        except (KeyboardInterrupt, EOFError):
            print()


if __name__ == "__main__":
    main()
