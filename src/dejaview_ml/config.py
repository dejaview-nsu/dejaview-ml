from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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
    model_config = SettingsConfigDict(
        env_prefix="ML_",
        env_file=".env",
        extra="ignore",
        env_parse_none_str="none",
        env_ignore_empty=True,
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True
    qdrant_timeout_seconds: int = Field(default=2, ge=1)
    qdrant_api_key: SecretStr | None = None
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
    # Порог релевантности: точки с похожестью ниже порога в ответ не попадают, если не осталось
    # ни одной - ответ «ничего не найдено». У модальностей своя шкала похожести, поэтому и свой
    # порог. Значения подобраны по выборке #17824, #17825, #18245 
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    score_threshold_text: float | None = Field(default=None, ge=-1, le=1)
    score_threshold_image: float | None = Field(default=0.8614, ge=-1, le=1)
    score_threshold_video: float | None = Field(default=0.9743, ge=-1, le=1)
    score_threshold_audio: float | None = Field(default=0.8297, ge=-1, le=1)
    ffmpeg_timeout_seconds: float = Field(default=10, gt=0)
    body_read_timeout_seconds: float = Field(default=30, gt=0)
    torch_threads: int = Field(default=4, ge=1)

    def threshold(self, modality: str) -> float | None:
        name = f"score_threshold_{modality}"
        if name in self.model_fields_set or "score_threshold" not in self.model_fields_set:
            return getattr(self, name)
        return self.score_threshold

    def enabled(self, modality: str) -> bool:
        return bool(getattr(self, f"{modality}_enabled", False))
