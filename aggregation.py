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


GEOMETRIC_FEATURE_DIM = 32
SEMANTIC_LAYER_COUNT = 2


def _masked_mean(tensor: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
    """tensor: (seq, dim), mask_seq: (seq,) bool."""
    mask_seq = mask_seq.to(device=tensor.device)
    w = mask_seq.float().unsqueeze(-1)
    denom = w.sum().clamp(min=1e-6)
    return (tensor * w).sum(dim=0) / denom


def _masked_scalar_mean(values: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
    mask_seq = mask_seq.to(device=values.device)
    w = mask_seq.float()
    denom = w.sum().clamp(min=1e-6)
    return ((values * w).sum() / denom).reshape(1)


def _masked_scalar_std(values: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
    mask_seq = mask_seq.to(device=values.device)
    mean = _masked_scalar_mean(values, mask_seq)
    w = mask_seq.float()
    denom = w.sum().clamp(min=1e-6)
    var = (((values - mean) ** 2) * w).sum() / denom
    return var.sqrt().reshape(1)


def _masked_variance_mean(tensor: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
    mask_seq = mask_seq.to(device=tensor.device)
    mean = _masked_mean(tensor, mask_seq)
    w = mask_seq.float().unsqueeze(-1)
    denom = w.sum().clamp(min=1e-6)
    var = (((tensor - mean) ** 2) * w).sum(dim=0) / denom
    return var.mean().reshape(1)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=1
    ).reshape(1)


def _layer(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
    if layer_idx < 0:
        return h[layer_idx]
    return h[min(layer_idx, h.size(0) - 1)]


def _response_and_context_masks(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate response tokens as the final valid span of the sequence."""
    valid_positions = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    n_valid = int(valid_positions.numel())
    tail_len = max(1, min(n_valid, int(round(n_valid * 0.40))))
    tail_start = valid_positions[-tail_len]

    response_mask = mask & (torch.arange(mask.numel(), device=mask.device) >= tail_start)
    context_mask = mask & ~response_mask
    if not bool(context_mask.any().item()):
        context_mask = response_mask
    return response_mask, context_mask


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compact hybrid representation from middle-to-late layers.

    Layer index convention (Hugging Face causal LMs):

    ``0`` embeddings, ``1…L`` outputs after transformer blocks ``1…L`` (here L = 24).

    Block A is a small geometric/statistical summary over layers around 9-18.
    Block B is semantic response-tail mean pooling from two strong layers. For
    Qwen-0.5B this yields 32 + 2 * 896 = 1824 features.
    """
    h = hidden_states.float()
    mask = attention_mask.reshape(-1).bool().to(device=h.device)
    hidden_dim = h.size(-1)

    real_count = int(mask.long().sum().item())
    if real_count == 0:
        return torch.zeros(
            GEOMETRIC_FEATURE_DIM + SEMANTIC_LAYER_COUNT * hidden_dim,
            dtype=h.dtype,
            device=h.device,
        )

    response_mask, context_mask = _response_and_context_masks(mask)
    n_valid = mask.float().sum().clamp(min=1.0)
    n_response = response_mask.float().sum().clamp(min=1.0)
    seq_len = torch.tensor(mask.numel(), dtype=h.dtype, device=h.device).clamp(min=1.0)

    stat_layers = [9, 12, 15, 18]
    response_means: list[torch.Tensor] = []
    context_means: list[torch.Tensor] = []
    geom: list[torch.Tensor] = [
        (n_valid / seq_len).reshape(1),
        (n_response / n_valid).reshape(1),
        torch.log(n_valid / seq_len + 1e-6).reshape(1),
    ]

    for layer_idx in stat_layers:
        layer = _layer(h, layer_idx)
        resp_mean = _masked_mean(layer, response_mask)
        ctx_mean = _masked_mean(layer, context_mask)
        response_means.append(resp_mean)
        context_means.append(ctx_mean)

        norms = layer.norm(p=2, dim=-1)
        geom.append(_masked_scalar_mean(norms, response_mask))
        geom.append(_masked_scalar_std(norms, response_mask))
        geom.append(_masked_scalar_mean(norms, context_mask))
        geom.append(_cosine(resp_mean, ctx_mean))
        geom.append(_masked_variance_mean(layer, response_mask))

    scale = hidden_dim**0.5
    for left, right in zip(response_means, response_means[1:]):
        drift = right - left
        geom.append(_cosine(left, right))
        geom.append((drift.norm(p=2) / scale).reshape(1))

    final_resp = response_means[-1]
    for prev in response_means[:-1]:
        geom.append(_cosine(prev, final_resp))

    geom_features = torch.cat(geom, dim=0)

    semantic_layers = [13, 18]
    semantic = [
        _masked_mean(_layer(h, layer_idx), response_mask)
        for layer_idx in semantic_layers
    ]

    return torch.cat([geom_features, *semantic], dim=0)


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