"""Shared sampling for the video index and search queries."""

VIDEO_SEGMENT_SEC = 1
VIDEO_SEGMENT_FRAMES = 1
VIDEO_FRAME_SIZE = 224
VIDEO_FILTER = (
    "fps=1:round=up,scale=224:224:force_original_aspect_ratio=increase:flags=bicubic,crop=224:224"
)
