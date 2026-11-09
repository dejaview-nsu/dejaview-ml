import logging
import re
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from dejaview_ml.config import DIMENSIONS, MODEL_IDS, SPEECH_MODEL_ID, Settings
from dejaview_ml.errors import MlError
from dejaview_ml.media import decode_audio, decode_image, decode_video, validate_text

logger = logging.getLogger(__name__)

AUDIO_RATE = 48_000
# Besides the whole clip, an audio query is searched in shorter overlapping windows, so that
# matches can be checked for consistency in time. Fewer than two windows give nothing to check.
AUDIO_QUERY_WINDOW_SEC = 5.0
AUDIO_QUERY_STEP_SEC = 2.5
VIDEO_QUERY_SEGMENT_SEC = 10.0
SPEECH_RATE = 16_000
SPEECH_MAX_TOKENS = 128
SPEECH_MIN_WORDS = 3
SPEECH_QUERY_CHARS = 250  # same limit as a typed text query
# Whisper fills silence and music with phrases it saw at the end of subtitled videos.
SPEECH_HALLUCINATIONS = re.compile(
    r"субтитр|продолжение следует|спасибо за (?:просмотр|внимание)|подписывайтесь|amara\.org",
    re.IGNORECASE,
)


def downsample_to_speech_rate(waveform: np.ndarray) -> np.ndarray:
    """48 kHz -> 16 kHz: windowed-sinc low-pass below the new Nyquist, then every third sample."""
    factor = AUDIO_RATE // SPEECH_RATE
    taps = np.arange(-48, 49)
    kernel = np.sinc(taps / factor) / factor * np.hamming(len(taps))
    return np.convolve(waveform, kernel, mode="same")[::factor].astype(np.float32)


def meaningful_speech(text: str) -> bool:
    return len(re.findall(r"\w+", text)) >= SPEECH_MIN_WORDS and not SPEECH_HALLUCINATIONS.search(
        text
    )


@dataclass(frozen=True)
class AudioQuery:
    """CLAP vectors with their start in the query, plus the recognised speech (if any)."""

    vectors: list[list[float]]
    offsets: list[float]
    transcript: str
    speech_vector: list[float] | None


class Encoders:
    """Real encoders, loaded once; shared by HTTP and indexing CLI."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.states = {
            kind: "loading" if settings.enabled(kind) else "disabled" for kind in DIMENSIONS
        }
        speech = settings.speech_enabled and settings.text_enabled and settings.audio_enabled
        self.states["speech"] = "loading" if speech else "disabled"
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

            if modality in {"video", "audio", "speech"} and not all(
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
            if modality == "speech":
                self._load_speech()
                self.states[modality] = "ready"
                logger.info(
                    "model ready modality=speech device=%s revision=%s",
                    self.device,
                    self.settings.speech_revision,
                )
                return
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

    def _load_speech(self) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        options = {
            "revision": self.settings.speech_revision,
            "cache_dir": str(self.settings.cache_dir),
            "local_files_only": self.settings.local_files_only,
        }
        processor = WhisperProcessor.from_pretrained(SPEECH_MODEL_ID, **options)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        model = WhisperForConditionalGeneration.from_pretrained(
            SPEECH_MODEL_ID, torch_dtype=dtype, **options
        )
        if processor.feature_extractor.sampling_rate != SPEECH_RATE:
            raise RuntimeError("expected Whisper with 16 kHz audio input")
        self.models["speech"] = model.to(self.device).eval()
        self.processors["speech"] = processor
        self._transcribe(np.zeros(AUDIO_RATE, dtype=np.float32))

    def _transcribe(self, waveform: np.ndarray) -> str:
        """Text of a 48 kHz mono clip, or "" when Whisper heard no meaningful speech."""
        import torch

        processor, model = self.processors["speech"], self.models["speech"]
        features = processor(
            downsample_to_speech_rate(waveform), sampling_rate=SPEECH_RATE, return_tensors="pt"
        ).input_features.to(self.device, model.dtype)
        with torch.inference_mode():
            tokens = model.generate(
                features,
                language=self.settings.speech_language,
                task="transcribe",
                max_new_tokens=SPEECH_MAX_TOKENS,
            )
        text = " ".join(processor.batch_decode(tokens, skip_special_tokens=True)[0].split())
        return text if meaningful_speech(text) else ""

    def transcribe_segments(self, chunks: list[np.ndarray]) -> list[list[tuple[float, float, str]]]:
        """Indexing: timed segments (start, end within the chunk, text) of 16 kHz chunks up to
        30 s each, transcribed as one batch. Whisper's end-of-video phrases are dropped."""
        import torch

        processor, model = self.processors["speech"], self.models["speech"]
        inputs = processor(
            chunks, sampling_rate=SPEECH_RATE, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            tokens = model.generate(
                inputs.input_features.to(self.device, model.dtype),
                attention_mask=inputs.attention_mask.to(self.device),
                language=self.settings.speech_language,
                task="transcribe",
                return_timestamps=True,
            )
        result = []
        for chunk, decoded in zip(
            chunks,
            processor.batch_decode(tokens, skip_special_tokens=True, output_offsets=True),
            strict=True,
        ):
            length = len(chunk) / SPEECH_RATE
            segments = []
            for segment in decoded["offsets"]:
                text = " ".join(segment["text"].split())
                start, end = segment["timestamp"]
                if text and not SPEECH_HALLUCINATIONS.search(text):
                    segments.append((min(start, length), min(end or length, length), text))
            result.append(segments)
        return result

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

    def encode_query(
        self, modality: str, data: bytes, content_type: str
    ) -> tuple[list[list[float]], list[float]]:
        """Vectors of a search query plus the start of each one inside the query, in seconds."""
        if modality != "audio":
            vectors = self.encode(modality, data, content_type)
            step = VIDEO_QUERY_SEGMENT_SEC if modality == "video" else 0.0
            return vectors, [index * step for index in range(len(vectors))]
        if self.states.get(modality) != "ready":
            raise MlError(503, "NOT_READY", f"model {modality} is {self.states.get(modality)}")
        waveform = decode_audio(data, content_type, self.settings.ffmpeg_timeout_seconds)
        return self._encode_audio_windows(waveform)

    def encode_audio_query(self, data: bytes, content_type: str) -> AudioQuery:
        """CLAP windows as in encode_query, plus Whisper transcript encoded by the text model.

        Speech is skipped (empty transcript) while the speech model is not ready, so audio
        search keeps working on CLAP alone.
        """
        if self.states.get("audio") != "ready":
            raise MlError(503, "NOT_READY", f"model audio is {self.states.get('audio')}")
        waveform = decode_audio(data, content_type, self.settings.ffmpeg_timeout_seconds)
        vectors, offsets = self._encode_audio_windows(waveform)
        transcript, speech_vector = "", None
        if self.states.get("speech") == "ready" and self.states.get("text") == "ready":
            transcript = self._transcribe(waveform)
            if transcript:
                speech_vector = self._encode_text(transcript[:SPEECH_QUERY_CHARS], passage=False)
        return AudioQuery(vectors, offsets, transcript, speech_vector)

    def _encode_audio_windows(self, waveform: np.ndarray) -> tuple[list[list[float]], list[float]]:
        windows = [(0.0, waveform)]
        size, step = (
            int(AUDIO_QUERY_WINDOW_SEC * AUDIO_RATE),
            int(AUDIO_QUERY_STEP_SEC * AUDIO_RATE),
        )
        if len(waveform) >= size + step:
            windows += [
                (start / AUDIO_RATE, waveform[start : start + size])
                for start in range(0, len(waveform) - size + 1, step)
            ]
        return [self._encode_audio(part) for _, part in windows], [offset for offset, _ in windows]
