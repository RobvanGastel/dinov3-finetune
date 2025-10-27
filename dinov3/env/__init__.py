# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import contextlib
import logging
import os
import tempfile
from typing import Optional

import submitit.helpers

logger = logging.getLogger("dinov3")


@contextlib.contextmanager
def clean_env():
    try:
        # Hide torch.compile() variables from the launched evals
        extra_names = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
        ctx = submitit.helpers.clean_env(extra_names=extra_names)
    except TypeError as e:
        logger.warning("Update submitit to the latest main branch\n%s", e)
        ctx = submitit.helpers.clean_env()
    with ctx:
        yield


def set_triton_cache_dir(cache_dir: Optional[str] = None) -> None:
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp()
    os.environ["TRITON_CACHE_DIR"] = cache_dir
