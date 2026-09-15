"""Fetch WeatherNext 2 weights and config from the public GCS bucket.

Weights: ``gs://dm_graphcast/weathernext2/params/{name}_<{split}_{ckpt}.npz``
(0.68 GiB each). Configs are resolved by name through
``weathernext.utils.fiddle_config_io``. Anonymous GCS access is enough.

On a GPU backend the attention implementation must be switched to
``triblockdiag_mha`` (slower and more memory hungry; H100 80 GB required
for the 0.25 deg models).
"""

from __future__ import annotations

from pathlib import Path


def download_checkpoint(bucket: str, model_name: str, split: str, checkpoint: str, cache_dir: Path) -> Path:
    """Download (or reuse) one weights file.

    Args:
        bucket: GCS bucket name (``dm_graphcast``).
        model_name: ``WeatherNext2`` or ``WeatherNextCyclones``.
        split: Training cutoff label, e.g. "2025".
        checkpoint: ``model1`` .. ``model4``.
        cache_dir: Local cache directory.

    Returns:
        Path to the ``.npz`` file.
    """
    raise NotImplementedError("TODO: anonymous GCS download with caching")


def build_predictor(config_name: str, ckpt_path: Path, gpu_attention_type: str):
    """Construct the jitted forward function and load parameters.

    Args:
        config_name: e.g. ``weathernext2/configs/WeatherNext2``.
        ckpt_path: Weights file from :func:`download_checkpoint`.
        gpu_attention_type: Attention override applied when backend is GPU.

    Returns:
        Tuple of (task config, jitted forward fn, params).
    """
    raise NotImplementedError("TODO: mirror docs/weathernext2/wn2_demo.ipynb")
