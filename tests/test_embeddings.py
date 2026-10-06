import numpy as np
import pytest
from PIL import Image

from dejaview_ml.config import MODEL_IDS, Settings
from dejaview_ml.embeddings import Encoders


@pytest.mark.parametrize("modality", ["image", "video"])
def test_load_full_clip_checkpoint_as_vision_encoder(tmp_path, monkeypatch, modality):
    """A full CLIP checkpoint must retain its trained vision and projection weights."""
    import torch
    import torch.nn.functional as functional
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

    torch.manual_seed(42)
    config = CLIPConfig(
        projection_dim=512,
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "projection_dim": 512,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 224,
        },
    )
    reference = CLIPModel(config).eval()
    reference.save_pretrained(tmp_path)
    processor = CLIPImageProcessor()
    processor.save_pretrained(tmp_path)
    monkeypatch.setitem(MODEL_IDS, modality, str(tmp_path))
    monkeypatch.setattr("dejaview_ml.embeddings.shutil.which", lambda executable: executable)

    encoders = Encoders(Settings(_env_file=None, device="cpu", local_files_only=True))
    encoders.load(modality)
    frame = Image.new("RGB", (224, 224), "orange")
    with torch.inference_mode():
        inputs = processor(images=[frame], return_tensors="pt")
        pooled = reference.vision_model(pixel_values=inputs.pixel_values).pooler_output
        expected = functional.normalize(reference.visual_projection(pooled), dim=-1)[0].numpy()
    actual = encoders._encode_frames(modality, [frame])
    assert encoders.states[modality] == "ready"
    assert len(actual) == 512
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_e5_loading_prefixes_masked_pooling_and_offline_reload(tmp_path, monkeypatch):
    import torch
    import torch.nn.functional as functional
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "query": 2,
                "passage": 3,
                ":": 4,
                "hello": 5,
                "world": 6,
                "warmup": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    processor = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    processor.save_pretrained(tmp_path)
    torch.manual_seed(42)
    model = XLMRobertaModel(
        XLMRobertaConfig(
            vocab_size=8,
            hidden_size=768,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=12,
            max_position_embeddings=514,
        )
    ).eval()
    model.save_pretrained(tmp_path)
    monkeypatch.setitem(MODEL_IDS, "text", str(tmp_path))
    settings = Settings(_env_file=None, text_enabled=True, device="cpu", local_files_only=True)
    encoders = Encoders(settings)
    encoders.load("text")

    # Force masked padding tokens into the batch to test average pooling as well as prefixes.
    class PaddedTokenizer:
        def __call__(self, texts, **kwargs):
            kwargs.update(padding="max_length", max_length=12)
            return processor(texts, **kwargs)

    encoders.processors["text"] = PaddedTokenizer()
    for passage, prefix in [(False, "query: "), (True, "passage: ")]:
        inputs = processor([prefix + "hello world"], return_tensors="pt")
        with torch.inference_mode():
            expected = functional.normalize(model(**inputs).last_hidden_state.mean(dim=1), dim=-1)
        actual = encoders.encode_text("hello world", passage=passage)[0]
        assert len(actual) == 768
        np.testing.assert_allclose(actual, expected[0].numpy(), atol=1e-6)
    # Index descriptions can exceed the search query's 250-character limit.
    assert len(encoders.encode_text("hello " * 100, passage=True)[0]) == 768
    reloaded = Encoders(settings)
    reloaded.load("text")
    np.testing.assert_allclose(
        reloaded.encode_text("hello world"), encoders.encode_text("hello world"), atol=1e-6
    )


def test_fused_clap_loading_projection_and_offline_reload(tmp_path, monkeypatch):
    import torch
    import torch.nn.functional as functional
    from transformers import ClapConfig, ClapFeatureExtractor, ClapModel

    torch.manual_seed(42)
    config = ClapConfig(
        projection_dim=512,
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 20,
        },
        audio_config={
            "hidden_size": 16,
            "patch_embeds_hidden_size": 8,
            "depths": [1, 1],
            "num_attention_heads": [2, 4],
            "num_hidden_layers": 2,
            "enable_fusion": True,
            "projection_dim": 512,
        },
    )
    reference = ClapModel(config).eval()
    reference.save_pretrained(tmp_path)
    processor = ClapFeatureExtractor()
    processor.save_pretrained(tmp_path)
    monkeypatch.setitem(MODEL_IDS, "audio", str(tmp_path))
    monkeypatch.setattr("dejaview_ml.embeddings.shutil.which", lambda executable: executable)
    settings = Settings(_env_file=None, audio_enabled=True, device="cpu", local_files_only=True)
    encoders = Encoders(settings)
    encoders.load("audio")
    waveform = np.sin(np.arange(48000, dtype=np.float32) * 0.06)
    with torch.inference_mode():
        inputs = processor(
            waveform,
            sampling_rate=48000,
            return_tensors="pt",
            truncation="fusion",
            padding="repeatpad",
        )
        pooled = reference.audio_model(**inputs).pooler_output
        expected = functional.normalize(reference.audio_projection(pooled), dim=-1)[0].numpy()
    actual = encoders._encode_audio(waveform)
    assert encoders.states["audio"] == "ready"
    assert len(actual) == 512
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    reloaded = Encoders(settings)
    reloaded.load("audio")
    np.testing.assert_allclose(reloaded._encode_audio(waveform), actual, atol=1e-6)
