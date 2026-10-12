import logging
import shutil
from typing import Any

import numpy as np
from PIL import Image

from dejaview_ml.config import DIMENSIONS, MODEL_IDS, Settings
from dejaview_ml.errors import MlError
from dejaview_ml.media import decode_audio, decode_image, decode_video, validate_text

logger = logging.getLogger(__name__)


class Encoders:
    """Real encoders, loaded once; shared by HTTP and indexing CLI."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.states = {
            kind: "loading" if settings.enabled(kind) else "disabled" for kind in DIMENSIONS
        }
        self.models: dict[str, Any] = {}
        self.processors: dict[str, Any] = {}
        self.device = "cpu"

    def load(self, modality: str) -> None:
        try:
            import torch
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionConfig,
                CLIPVisionModelWithProjection,
            )

            if modality in {"video", "audio"} and not all(
                shutil.which(x) for x in ("ffmpeg", "ffprobe")
            ):
                raise RuntimeError("ffmpeg and ffprobe are required for video/audio")
            torch.set_num_threads(self.settings.torch_threads)
            self.device = self.settings.device
            if self.device == "auto":
                self.device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu")
                )
            options = {
                "revision": getattr(self.settings, f"{modality}_revision"),
                "cache_dir": str(self.settings.cache_dir),
                "local_files_only": self.settings.local_files_only,
            }
            repo = MODEL_IDS[modality]
            if modality == "text":
                from transformers import AutoModel, AutoTokenizer

                processor = AutoTokenizer.from_pretrained(repo, **options)
                model = AutoModel.from_pretrained(repo, **options)
                dimension = model.config.hidden_size
            elif modality == "audio":
                from transformers import (
                    ClapAudioConfig,
                    ClapAudioModelWithProjection,
                    ClapFeatureExtractor,
                )

                processor = ClapFeatureExtractor.from_pretrained(repo, **options)
                audio_config = ClapAudioConfig.from_pretrained(repo, **options)
                model = ClapAudioModelWithProjection.from_pretrained(
                    repo,
                    config=audio_config,
                    **options,
                )
                dimension = model.config.projection_dim
                if processor.sampling_rate != 48_000 or not audio_config.enable_fusion:
                    raise RuntimeError("expected fused CLAP with 48 kHz audio input")
            else:
                processor = CLIPImageProcessor.from_pretrained(repo, **options)
                # Explicit subconfig avoids CLIPConfig being inferred for a vision-only model.
                vision_config = CLIPVisionConfig.from_pretrained(repo, **options)
                model = CLIPVisionModelWithProjection.from_pretrained(
                    repo,
                    config=vision_config,
                    **options,
                )
                dimension = model.config.projection_dim
            if dimension != DIMENSIONS[modality]:
                raise RuntimeError(f"unexpected embedding dimension for {modality}")
            self.models[modality] = model.to(self.device).eval()
            self.processors[modality] = processor
            if modality == "text":
                self._encode_text("warmup", passage=False)
            elif modality == "audio":
                self._encode_audio(np.zeros(48_000, dtype=np.float32))
            else:
                self._encode_frames(modality, [Image.new("RGB", (224, 224))])
            self.states[modality] = "ready"
            logger.info(
                "model ready modality=%s device=%s revision=%s",
                modality,
                self.device,
                options["revision"],
            )
        except Exception:
            self.states[modality] = "failed"
            raise

    def _encode_frames(self, modality: str, frames: list[Image.Image]) -> list[float]:
        import torch
        import torch.nn.functional as functional

        options = {"do_resize": False, "do_center_crop": False} if modality == "video" else {}
        inputs = self.processors[modality](images=frames, return_tensors="pt", **options)
        with torch.inference_mode():
            features = self.models[modality](pixel_values=inputs.pixel_values.to(self.device))
            frames_normalized = functional.normalize(features.image_embeds.float(), dim=-1)
            vector = functional.normalize(frames_normalized.mean(dim=0), dim=0).cpu().numpy()
        return self._validate_vector(modality, vector)

    @staticmethod
    def _validate_vector(modality: str, vector: np.ndarray) -> list[float]:
        if vector.shape != (DIMENSIONS[modality],) or not np.isfinite(vector).all():
            raise RuntimeError("model produced an invalid embedding")
        if float(np.linalg.norm(vector)) < 0.99:
            raise RuntimeError("model produced a zero embedding")
        return vector.astype(np.float32).tolist()

    def _encode_text(self, text: str, *, passage: bool) -> list[float]:
        import torch
        import torch.nn.functional as functional

        prefix = "passage: " if passage else "query: "
        inputs = self.processors["text"](
            [prefix + text],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.models["text"](**inputs).last_hidden_state.float()
            mask = inputs.attention_mask.unsqueeze(-1).bool()
            pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vector = functional.normalize(pooled, dim=-1)[0].cpu().numpy()
        return self._validate_vector("text", vector)

    def _encode_audio(self, waveform: np.ndarray) -> list[float]:
        import torch
        import torch.nn.functional as functional

        inputs = self.processors["audio"](
            waveform,
            sampling_rate=48_000,
            return_tensors="pt",
            truncation="fusion",
            padding="repeatpad",
        ).to(self.device)
        with torch.inference_mode():
            features = self.models["audio"](**inputs).audio_embeds.float()
            vector = functional.normalize(features, dim=-1)[0].cpu().numpy()
        return self._validate_vector("audio", vector)

    def encode_text(self, text: str, *, passage: bool = False) -> list[list[float]]:
        if self.states["text"] != "ready":
            raise MlError(503, "NOT_READY", f"model text is {self.states['text']}")
        return [self._encode_text(validate_text(text, passage=passage), passage=passage)]

    def encode(self, modality: str, data: bytes, content_type: str) -> list[list[float]]:
        if self.states.get(modality) != "ready":
            raise MlError(503, "NOT_READY", f"model {modality} is {self.states.get(modality)}")
        if modality == "image":
            return [self._encode_frames(modality, [decode_image(data, content_type)])]
        if modality == "audio":
            waveform = decode_audio(data, content_type, self.settings.ffmpeg_timeout_seconds)
            return [self._encode_audio(waveform)]
        if modality != "video":
            raise ValueError("use encode_text for text input")
        segments = decode_video(data, content_type, self.settings.ffmpeg_timeout_seconds)
        return [self._encode_frames(modality, frames) for frames in segments]
