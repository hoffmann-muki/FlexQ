import os
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

def download_llama2_7b_no_cache(
    token: str,
    save_dir: str = "models/llama-2-7b-hf"
):
    """
    Download LLaMA-2-7B-HF *without* using the HuggingFace cache.
    """

    # Force all HF Hub cache operations to use a temp directory
    # that will be removed automatically if not saved
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = "/tmp/hf_home_no_cache"
    os.environ["HF_HUB_CACHE"] = "/tmp/hf_cache_no_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache_no_cache"

    logger.info("Downloading LLaMA-2-7B-HF without using HuggingFace cache...")

    snapshot_download(
        repo_id="meta-llama/Llama-2-7b-hf",
        token=token,
        local_dir=save_dir,
        local_dir_use_symlinks=False,  # ensures files are copied, not symlinked
        allow_patterns="*"            # ensures only actual model files are saved
    )

    logger.info(f"Download complete. Model stored in: {save_dir}")


if __name__ == "__main__":
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("Set HUGGINGFACE_TOKEN first.")

    download_llama2_7b_no_cache(token)
