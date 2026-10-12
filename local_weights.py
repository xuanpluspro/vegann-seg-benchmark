"""Local SMP 0.5.0 ImageNet encoder weights; no network during training."""
from pathlib import Path
from safetensors.torch import load_file

def load_local_encoder(model, encoder, directory):
    path = Path(directory) / f"{encoder}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Missing local encoder weights: {path}. Run download_pretrained.py on a networked computer and upload pretrained/.")
    state = load_file(str(path), device="cpu")
    model.encoder.load_state_dict(state, strict=True)
    print(f"Loaded local ImageNet encoder: {path}", flush=True)
