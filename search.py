import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_ts(seconds):
    """3725.0 -> '1:02:05'"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@torch.no_grad()
def embed_text(text, model, processor, device):
    """Turn a sentence into a unit-length 512-number vector in the SAME space as the images."""
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")   # shape (1, 512)


def search(query, index, timestamps, model, processor, device, top=5, gap=10.0):
    q = embed_text(query, model, processor, device)

    # Ask FAISS for many more candidates than we need, because neighbouring
    # seconds of a lecture look almost identical (same slide on screen).
    # Without this, the "top 5" would often be 0:41, 0:42, 0:43, 0:44, 0:45.
    k = min(index.ntotal, top * 20)
    scores, ids = index.search(q, k)       # both shape (1, k), best first

    results = []
    for score, i in zip(scores[0], ids[0]):
        ts = float(timestamps[i])
        # Skip this hit if it's within `gap` seconds of one we already kept.
        if all(abs(ts - kept_ts) >= gap for kept_ts, _ in results):
            results.append((ts, float(score)))
        if len(results) == top:
            break
    return results


def main():
    parser = argparse.ArgumentParser(description="Search a lecture index with text.")
    parser.add_argument("index_dir", help="Folder created by index_video.py")
    parser.add_argument("query", nargs="?", help="Text to search for (omit for interactive mode)")
    parser.add_argument("--top", type=int, default=5, help="How many results to print")
    parser.add_argument("--gap", type=float, default=10.0,
                        help="Min seconds between results (0 = allow near-duplicates)")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "frames.faiss"))
    timestamps = np.load(index_dir / "timestamps.npy")
    meta = json.loads((index_dir / "meta.json").read_text())

    # MUST be the same model that built the index, or the vectors live in
    # different "spaces" and the comparison is meaningless.
    device = pick_device()
    model = CLIPModel.from_pretrained(meta["model"]).to(device).eval()
    processor = CLIPProcessor.from_pretrained(meta["model"])

    print(f"Loaded {index.ntotal} frames from {Path(meta['video']).name}")

    def run(query):
        print(f'\nTop {args.top} for: "{query}"')
        for rank, (ts, score) in enumerate(
                search(query, index, timestamps, model, processor, device, args.top, args.gap), 1):
            print(f"  {rank}. {format_ts(ts):>8}   (score {score:.3f})")

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
