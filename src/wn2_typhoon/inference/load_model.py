"""Fetch WeatherNext 2 weights and build the jitted forward function.

Weights live at ``gs://dm_graphcast/weathernext2/params/``; configs are
resolved by name through ``weathernext.utils.fiddle_config_io`` and ship inside
the package, so the task config can be read without touching the network.

The attention implementation has to be chosen per backend. All bundled configs
default to ``splash_mha``, which is the TPU Pallas kernel; the upstream demo
overrides it for GPU only, so CPU runs need the same treatment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wn2_typhoon.inference.gcs import download_blob

PARAMS_PREFIX = "weathernext2/params"

# jax backend -> attention_type. Backends absent from this table keep whatever
# the checkpoint config specifies.
ATTENTION_TYPE_BY_BACKEND = {
    "cpu": "triblockdiag_mha",
    "gpu": "triblockdiag_mha",
}


def checkpoint_blob_name(model_name: str, split: str, checkpoint: str | None) -> str:
    """Build the GCS object path of one weights file.

    Args:
        model_name: e.g. "WeatherNext2" or "WeatherNextCyclones_Mini".
        split: Training cutoff label, e.g. "2025".
        checkpoint: "model1" .. "model4" for the 0.25 deg models. The Mini
            models ship a single checkpoint and take None.

    Returns:
        Object path within the bucket.
    """
    suffix = f"_{checkpoint}" if checkpoint else ""
    return f"{PARAMS_PREFIX}/{model_name}_<{split}{suffix}.npz"


def download_checkpoint(
    bucket: str,
    model_name: str,
    split: str,
    checkpoint: str | None,
    cache_dir: Path,
) -> Path:
    """Download (or reuse) one weights file.

    Args:
        bucket: GCS bucket name ("dm_graphcast").
        model_name: See :func:`checkpoint_blob_name`.
        split: Training cutoff label.
        checkpoint: Checkpoint label, or None for the Mini models.
        cache_dir: Local cache directory.

    Returns:
        Path to the ``.npz`` file.
    """
    blob_name = checkpoint_blob_name(model_name, split, checkpoint)
    return download_blob(bucket, blob_name, cache_dir)


def attention_type_for_backend(backend: str) -> str | None:
    """Return the attention implementation to force for a jax backend.

    Args:
        backend: Value of ``jax.default_backend()``.

    Returns:
        The ``attention_type`` to set, or None to keep the config default.
    """
    return ATTENTION_TYPE_BY_BACKEND.get(backend)


def build_predictor(config_name: str, ckpt_path: Path, backend: str) -> tuple:
    """Load weights and build the pmapped forward function.

    Mirrors ``docs/weathernext2/wn2_demo.ipynb``: the ensemble wrapper is
    dropped from the predictor so that members are drawn by folding the RNG
    key, and the result is pmapped over the ``sample`` dimension.

    Args:
        config_name: e.g. "weathernext2/configs/WeatherNext2".
        ckpt_path: Weights file from :func:`download_checkpoint`.
        backend: Value of ``jax.default_backend()``; selects the attention
            implementation via :func:`attention_type_for_backend`.

    Returns:
        Tuple of (task config, pmapped forward function).
    """
    import haiku as hk
    import jax
    import xarray_jax
    from weathernext.utils import checkpoint, fiddle_config_io
    from weathernext.weathernext2 import fgn

    config = fiddle_config_io.get_fiddle_config_by_name(config_name)

    attention_type = attention_type_for_backend(backend)
    if attention_type is not None:
        transformer_kwargs = config.predictor_kwargs["noisy_function_kwargs"][
            "mesh_model_ctor"
        ].keywords["transformer_kwargs"]
        transformer_kwargs["attention_type"] = attention_type

    config_inference = fgn.PredictorConfig(
        task=config.task,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        # The last wrapper draws an ensemble internally; members come from
        # folded RNG keys instead, one device at a time.
        predictor_wrappers=config.predictor_wrappers[:-1],
    )

    @hk.transform
    def run_forward(inputs: Any, targets_template: Any, forcings: Any) -> Any:
        predictor = fgn.construct_predictor(config_inference)
        return predictor(
            inputs, targets_template=targets_template, forcings=forcings
        )

    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, fgn.CheckPoint)

    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(ckpt.params, rng, i, t, f)
    )
    return config.task, xarray_jax.pmap(run_forward_jitted, dim="sample")
