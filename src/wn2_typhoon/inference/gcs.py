"""Anonymous downloads from the public ``dm_graphcast`` bucket.

Weights and sample datasets are served without credentials. Everything is
cached on disk because the files are large (0.23-0.74 GiB for weights, up to
13 GiB for sample data) and the pod is billed by the hour.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_blob(bucket_name: str, blob_name: str, cache_dir: Path) -> Path:
    """Download one object, reusing an existing complete copy.

    The cached file keeps the object's base name. A size mismatch counts as an
    incomplete download and triggers a refetch; the bucket is versioned content
    that never changes in place, so size is a sufficient check.

    Args:
        bucket_name: GCS bucket, e.g. "dm_graphcast".
        blob_name: Object path within the bucket.
        cache_dir: Directory holding the local copies.

    Returns:
        Path to the local file.
    """
    from google.cloud import storage

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(blob_name).name

    client = storage.Client.create_anonymous_client()
    blob = client.bucket(bucket_name).get_blob(blob_name)
    if blob is None:
        raise FileNotFoundError(f"gs://{bucket_name}/{blob_name}")

    if dest.exists() and dest.stat().st_size == blob.size:
        logger.info("Using cached %s", dest)
        return dest

    logger.info("Downloading gs://%s/%s (%.1f MiB)", bucket_name, blob_name,
                blob.size / 1024**2)
    partial = dest.with_suffix(dest.suffix + ".part")
    blob.download_to_filename(str(partial))
    partial.replace(dest)
    return dest
