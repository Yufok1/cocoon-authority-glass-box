"""Cocoon Bucket Membrane — the artery between the two HF Spaces.

The Butterfly Space breeds organisms and exports cocoons into a shared HF bucket;
the Cocoon Space pulls them to train / compile, and pushes trained or bay-compiled
cocoons back. Pure `huggingface_hub` Python buckets API — identical locally and on
the Space. Auth via the `HF_TOKEN` env (set as a Space secret); falls back to the
cached/anonymous token locally.

This is the dynamic-association layer: generate on one Space, train on the other,
both read/write the same bucket.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import (
        list_bucket_tree, download_bucket_files, batch_bucket_files, bucket_info,
    )
    _HF_OK = True
    _IMPORT_ERR = None
except Exception as _e:  # never let a missing dep break the Authority import
    _HF_OK = False
    _IMPORT_ERR = str(_e)

# the shared bucket + the cocoon-exchange prefix (both overridable by env)
BUCKET_ID = os.environ.get("COCOON_BUCKET", "tostido/Butterfly-Field-Station-storage")
COCOON_PREFIX = os.environ.get("COCOON_PREFIX", "cocoon_bay").rstrip("/")
_ENV_TOKEN = os.environ.get("HF_TOKEN") or None


def available() -> bool:
    return _HF_OK


def _tok(token):
    return token if token is not None else _ENV_TOKEN


def status(token=None) -> dict:
    """Bucket health + cocoon count, for the induction/management menu header."""
    if not _HF_OK:
        return {"ok": False, "error": f"huggingface_hub buckets unavailable: {_IMPORT_ERR}"}
    try:
        info = bucket_info(BUCKET_ID, token=_tok(token))
        cocoons = list_cocoons(token=token)
        return {"ok": True, "bucket": BUCKET_ID, "prefix": COCOON_PREFIX,
                "total_files": getattr(info, "total_files", None),
                "size": getattr(info, "size", None), "cocoons": len(cocoons)}
    except Exception as e:
        return {"ok": False, "bucket": BUCKET_ID, "error": str(e)}


def list_cocoons(prefix: Optional[str] = None, token=None) -> list[dict]:
    """List cocoon `.py` superfiles under the bucket prefix (induction menu source)."""
    if not _HF_OK:
        return []
    pre = COCOON_PREFIX if prefix is None else prefix.rstrip("/")
    out: list[dict] = []
    for item in list_bucket_tree(BUCKET_ID, pre, recursive=True, token=_tok(token)):
        path = str(getattr(item, "path", item))
        if path.endswith(".py"):
            out.append({"path": path, "name": Path(path).name,
                        "size": getattr(item, "size", None)})
    return sorted(out, key=lambda c: c["name"])


def pull_cocoons(remote_paths: list[str], local_dir: str, token=None) -> list[str]:
    """Download cocoons from the bucket to a local dir (to train / compile)."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    files = [(r, str(Path(local_dir) / Path(r).name)) for r in remote_paths]
    download_bucket_files(BUCKET_ID, files, token=_tok(token))
    return [f[1] for f in files]


def pull_cocoon(remote_path: str, local_path: str, token=None) -> str:
    download_bucket_files(BUCKET_ID, [(remote_path, local_path)], token=_tok(token))
    return local_path


def push_cocoon(local_path: str, remote_name: Optional[str] = None,
                prefix: Optional[str] = None, token=None) -> str:
    """Upload a trained / bay-compiled cocoon back to the shared bucket."""
    pre = COCOON_PREFIX if prefix is None else prefix.rstrip("/")
    name = remote_name or Path(local_path).name
    remote = f"{pre}/{name}"
    batch_bucket_files(BUCKET_ID, add=[(local_path, remote)], token=_tok(token))
    return remote


def delete_cocoon(remote_path: str, token=None) -> None:
    batch_bucket_files(BUCKET_ID, delete=[remote_path], token=_tok(token))
