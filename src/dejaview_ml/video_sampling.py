"""Video index: 0.5 FPS; search queries: 1 FPS, with shared preprocessing."""

VIDEO_SEGMENT_SEC = 2
VIDEO_QUERY_SEGMENT_SEC = 1
VIDEO_SEGMENT_FRAMES = 1
VIDEO_FRAME_SIZE = 224
_PREPROCESS = "scale=224:224:force_original_aspect_ratio=increase:flags=bicubic,crop=224:224"
VIDEO_FILTER = f"fps=1/{VIDEO_SEGMENT_SEC}:round=up,{_PREPROCESS}"
VIDEO_QUERY_FILTER = f"fps=1/{VIDEO_QUERY_SEGMENT_SEC}:round=up,{_PREPROCESS}"
