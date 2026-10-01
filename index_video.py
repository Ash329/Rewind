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

# A small, fast CLIP model. Every vector it produces has 512 numbers.
MODEL_NAME = "openai/clip-vit-base-patch32"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"          # NVIDIA GPU
    if torch.backends.mps.is_available():
        return "mps"           # Apple Silicon GPU
    return "cpu"


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


def main():
    parser = argparse.ArgumentParser(description="Index a lecture video with CLIP + FAISS.")
    parser.add_argument("video", help="Path to the video file (mp4, mkv, ...)")
    parser.add_argument("--out", default=None, help="Output folder (default: index_<video name>)")
    parser.add_argument("--every", type=float, default=1.0, help="Seconds between sampled frames")
    parser.add_argument("--batch", type=int, default=32, help="Frames sent to CLIP at once")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out or f"index_{video_path.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    for ts, img in sample_frames(video_path, args.every):
        batch_imgs.append(img)
        batch_ts.append(ts)
        if len(batch_imgs) == args.batch:
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
    (out_dir / "meta.json").write_text(json.dumps({
        "video": str(video_path.resolve()),
        "model": MODEL_NAME,
        "every_seconds": args.every,
        "num_frames": int(index.ntotal),
    }, indent=2))

    print(f"Done: {index.ntotal} frames indexed in {time.time() - start:.0f}s -> {out_dir}/")


if __name__ == "__main__":
    main()
