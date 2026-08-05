"""PDF RAG application with grounded, cited answers."""

import os

# Keep CLI output clean: silence the HuggingFace/fastembed download progress bars
# and the tokenizers fork warning. Set before any submodule imports fastembed.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
