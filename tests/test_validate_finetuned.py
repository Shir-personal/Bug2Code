import types

import numpy as np
import pandas as pd
import torch
from torch import nn

from bug2code.localization import validate_finetuned
from bug2code.localization.validate_finetuned import FineTunedEncoder, select_best_epoch


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer, mirroring the fake used for training tests.

    Handles both call shapes ``FineTunedEncoder`` needs: a batch of texts with
    padding/truncation (``encode_texts``) and a single un-padded text with
    ``add_special_tokens=False`` (``encode_files``'s per-file tokenization).
    """

    cls_token_id = 0
    sep_token_id = 1
    pad_token_id = 2

    def __call__(
        self, texts, padding=None, truncation=None, max_length=None, add_special_tokens=True
    ):
        single = isinstance(texts, str)
        texts_list = [texts] if single else texts
        ids, masks = [], []
        for text in texts_list:
            tok_ids = [3 + (ord(c) % 5) for c in text]
            if max_length is not None:
                tok_ids = tok_ids[:max_length]
            ids.append(tok_ids)
            masks.append([1] * len(tok_ids))
        if single:
            return {"input_ids": ids[0], "attention_mask": masks[0]}
        if padding:
            max_len = max((len(i) for i in ids), default=0)
            for i in range(len(ids)):
                pad_len = max_len - len(ids[i])
                ids[i] = ids[i] + [self.pad_token_id] * pad_len
                masks[i] = masks[i] + [0] * pad_len
        return {"input_ids": ids, "attention_mask": masks}


class _FakeModel(nn.Module):
    """Tiny learnable stand-in for DualEncoder: same encode_query/encode_code
    interface and a query_encoder.config.hidden_size attribute, no pretrained
    weights, so tests stay fast and fully offline."""

    def __init__(self, vocab_size: int = 16, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.query_encoder = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=dim))

    def _pool(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(ids)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=1)

    def encode_query(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._pool(ids, mask)

    def encode_code(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._pool(ids, mask)


def _encoder(fp16: bool = False, device: str = "cpu") -> FineTunedEncoder:
    return FineTunedEncoder(_FakeModel(), _FakeTokenizer(), device, fp16=fp16)


def test_encode_texts_returns_l2_normalized_vectors_of_expected_shape():
    encoder = _encoder()
    vecs = encoder.encode_texts(["a bug report", "another one"], max_length=8, batch_size=2)
    assert vecs.shape == (2, 8)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_files_chunk_ranges_cover_all_chunks_contiguously():
    encoder = _encoder()
    cfg = types.SimpleNamespace(
        localization={"code_max_length": 8, "chunk_stride": 4, "max_chunks_per_file": 12}
    )
    texts = {10: "short text", 20: "a considerably longer piece of source code content here"}

    vecs, ranges = encoder.encode_files(texts, cfg, batch_size=4)

    assert set(ranges) == {10, 20}
    start10, end10 = ranges[10]
    start20, end20 = ranges[20]
    assert start10 == 0
    assert start20 == end10  # contiguous, no gaps or overlaps between files
    assert vecs.shape[0] == end20
    assert vecs.shape[1] == 8


def test_fp16_flag_is_ignored_off_cuda():
    encoder = _encoder(fp16=True, device="cpu")
    assert encoder.fp16 is False


def test_select_best_epoch_picks_highest_metric():
    summary = pd.DataFrame(
        {"mrr": [0.30, 0.41, 0.38]}, index=pd.Index([1, 2, 3], name="epoch")
    )
    assert select_best_epoch(summary) == 2


def test_select_best_epoch_uses_requested_metric():
    summary = pd.DataFrame(
        {"mrr": [0.41, 0.30, 0.38], "hit_at_1": [0.10, 0.50, 0.20]},
        index=pd.Index([1, 2, 3], name="epoch"),
    )
    assert select_best_epoch(summary, metric="hit_at_1") == 2


def test_load_finetuned_model_loads_both_towers_from_checkpoint(tmp_path, monkeypatch):
    class _TinyTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 2)

    class _TinyDualEncoder(nn.Module):
        def __init__(self, hf_name, dropout):
            super().__init__()
            self.query_encoder = _TinyTower()
            self.code_encoder = _TinyTower()

    monkeypatch.setattr(validate_finetuned, "DualEncoder", _TinyDualEncoder)

    source = _TinyDualEncoder("unused", 0.1)
    with torch.no_grad():
        source.query_encoder.w.weight.fill_(1.0)
        source.code_encoder.w.weight.fill_(2.0)
    ckpt_path = tmp_path / "epoch_1.pt"
    torch.save(
        {
            "query_encoder_state_dict": source.query_encoder.state_dict(),
            "code_encoder_state_dict": source.code_encoder.state_dict(),
        },
        ckpt_path,
    )

    loaded = validate_finetuned.load_finetuned_model(ckpt_path, dropout=0.1, device="cpu")

    assert torch.equal(loaded.query_encoder.w.weight, source.query_encoder.w.weight)
    assert torch.equal(loaded.code_encoder.w.weight, source.code_encoder.w.weight)
