import sys
from huggingface_hub import snapshot_download


# Usage:
#   python scripts/download_siglip_encoder.py
#   python scripts/download_siglip_encoder.py google/siglip2-large-patch16-256
#   python scripts/download_siglip_encoder.py google/siglip2-large-patch16-256 ./checkpoints/siglip
model_id = sys.argv[1] if len(sys.argv) >= 2 else "google/siglip2-large-patch16-256"
local_dir = sys.argv[2] if len(sys.argv) >= 3 else "./checkpoints/siglip2-large-patch16-256"

print(f"Downloading SigLIP encoder from {model_id} ...")
print(f"Saving to: {local_dir}")
print("HF endpoint: default (huggingface.co)")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
)

print("Done.")
