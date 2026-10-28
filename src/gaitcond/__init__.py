"""Walking-condition classification on CASIA-B gait silhouettes."""

CLASSES = ("nm", "bg", "cl")
CLASS_NAMES = {"nm": "normal", "bg": "bag", "cl": "coat"}
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

__version__ = "0.2.0"
