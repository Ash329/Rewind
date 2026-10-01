"""
Where to run the models. Phase 2 needs this in three places (frames, speech,
search), so it lives here instead of being copy-pasted.
"""

import torch


def pick_device():
    if torch.cuda.is_available():
        return "cuda"                           # NVIDIA GPU
    if torch.backends.mps.is_available():
        return "mps"                            # Apple Silicon GPU
    return "cpu"


def free_gpu():
    """
    Hand back GPU memory that nothing references any more.

    Call this right after `del`-ing a model you are finished with. Phase 2 runs
    CLIP and then Whisper on the same card, and on a 4 GB laptop GPU that is the
    difference between "works" and "CUDA out of memory" -- otherwise the CLIP
    weights sit there unused for the whole transcription.

    (The `del` has to happen at the call site: deleting a function argument only
    drops this function's own reference, not the caller's.)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
