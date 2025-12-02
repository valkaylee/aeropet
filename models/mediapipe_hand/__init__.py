# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from .app import MediaPipeHandApp as App  # noqa: F401
# from .model import MODEL_ID  # noqa: F401
# Note: MediaPipeHand is in model.py (file), not model/ (directory)
# Import it directly: from models.mediapipe_hand.model import MediaPipeHand
# Commented out to avoid import conflicts when importing from model subdirectory
# from .model import MediaPipeHand as Model  # noqa: F401
