from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DIMENSIONS = {"text": 768, "image": 512, "video": 512, "audio": 512}
MODEL_IDS = {
    "text": "intfloat/multilingual-e5-base",
    "image": "openai/clip-vit-base-patch32",
    "video": "Searchium-ai/clip4clip-webvid150k",
    "audio": "laion/clap-htsat-fused",
}
REVISIONS = {
    "text": "d128750597153bb5987e10b1c3493a34e5a4502a",
    "image": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "video": "d2f89850fec28ac1e5cd27db2f0b76d7491e61ce",
    "audio": "365dea6ef167def6676140ed93bbc43f84dabb28",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_", env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True
    qdrant_timeout_seconds: int = Field(default=2, ge=1)
    cache_dir: Path = Path(".cache/huggingface")
    local_files_only: bool = False
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    image_enabled: bool = True
    video_enabled: bool = True
    text_enabled: bool = True
    audio_enabled: bool = True
    text_revision: str = REVISIONS["text"]
    audio_revision: str = REVISIONS["audio"]
    image_revision: str = REVISIONS["image"]
    video_revision: str = REVISIONS["video"]
    top_k: int = Field(default=20, ge=1, le=1000)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    ffmpeg_timeout_seconds: float = Field(default=10, gt=0)
    torch_threads: int = Field(default=4, ge=1)

    def enabled(self, modality: str) -> bool:
        return bool(getattr(self, f"{modality}_enabled", False))
