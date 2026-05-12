"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.
"""

from __future__ import annotations

import os
import sys
import types


def _stub_model_maybe() -> None:
    """Registers a deterministic tiny LM shim so ``python solution.py`` can dry-run without a heavyweight forward.

    Activated only when ``SMILES_STUB_LM`` is truthy ("1"/"yes"/"true"). Leave unset for the competition.
    Setting this **before** importing ``solution`` lets this module preempt the real Hugging Face loader.
    """
    if os.getenv("SMILES_STUB_LM", "").strip().lower() not in {"1", "yes", "true"}:
        return

    already = sys.modules.setdefault("model", types.ModuleType("model"))
    if getattr(already, "_smiles_stub_injected", False):
        return

    try:
        import torch

        tokenizer_mod = sys.modules.setdefault(
            "transformers", __import__("transformers")
        )
        AutoTokenizer = tokenizer_mod.AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "SMILES_STUB_LM requires torch + transformers (same as the full pipeline)."
        ) from exc

    print("[SMILES_STUB_LM] Using deterministic stub causal LM — results are INVALID for submissions.")

    _MAX_LENGTH = 512

    class _StubTokenizer:
        """Minimal wrapper with the methods ``solution.py`` invokes."""

        def __init__(self, inner) -> None:
            self.inner = inner
            if getattr(self.inner, "pad_token", None) is None and getattr(
                self.inner, "eos_token", None
            ) is not None:
                self.inner.pad_token = self.inner.eos_token

        def __call__(self, texts, **kwargs):
            enc = self.inner(texts, **kwargs)

            ids = torch.as_tensor(enc["input_ids"])
            attn = torch.as_tensor(enc["attention_mask"])
            if kwargs.get("return_tensors") == "pt":
                return {"input_ids": ids, "attention_mask": attn}
            raise NotImplementedError

        def __getattr__(self, name):
            return getattr(self.inner, name)

    class _StubCausalLM(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, **_kwargs):
            device = input_ids.device
            b, seq = input_ids.shape
            hid = 896

            seed_mix = (
                int(input_ids.detach().to(dtype=torch.int64).sum().item()) % 1_000_007
            )
            scale = 6.5

            stacks: list[torch.Tensor] = []
            for blk in range(25):
                g_cpu = torch.Generator()
                g_cpu.manual_seed((seed_mix + blk * 7919) % (2**32))
                blk_t = torch.randn((b, seq, hid), generator=g_cpu) / scale
                stacks.append(blk_t.to(device=device, dtype=torch.bfloat16))

            return types.SimpleNamespace(hidden_states=tuple(stacks))

    def _stub_get_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B"):
        tok_inner = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        stub = _StubCausalLM()
        stub.train(False)
        return stub, _StubTokenizer(tok_inner)

    setattr(sys.modules["model"], "MAX_LENGTH", _MAX_LENGTH)
    setattr(sys.modules["model"], "get_model_and_tokenizer", _stub_get_model_and_tokenizer)
    setattr(sys.modules["model"], "_smiles_stub_injected", True)


_stub_model_maybe()


import torch


def _masked_mean(tensor: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
    """tensor: (seq, dim), mask_seq: (seq,) bool."""
    mask_seq = mask_seq.to(device=tensor.device)
    w = mask_seq.float().unsqueeze(-1)
    denom = w.sum().clamp(min=1e-6)
    return (tensor * w).sum(dim=0) / denom


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compact late-layer mean-pooled representation.

    Layer index convention (Hugging Face causal LMs):

    ``0`` embeddings, ``1…L`` outputs after transformer blocks ``1…L`` (here L = 24).

    The previous feature set mixed several pooling modes and cross-layer geometry,
    producing >12k dimensions for Qwen-0.5B.  For this small dataset we keep only
    mean-pooled hidden states from the final few transformer layers, which should
    yield 3 * hidden_size features (2688 for Qwen-0.5B).
    """
    h = hidden_states.float()
    mask = attention_mask.reshape(-1).bool().to(device=h.device)

    real_count = int(mask.long().sum().item())
    if real_count == 0:
        return torch.zeros(h.size(-1) * 3, dtype=h.dtype, device=h.device)

    late = (-3, -2, -1)
    feats = [_masked_mean(h[layer_idx], mask) for layer_idx in late]

    return torch.cat(feats, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    del attention_mask
    return torch.zeros(0, dtype=hidden_states.dtype, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features