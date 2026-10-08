"""Inverse of llama.cpp's Qwen3.5 HF->GGUF tensor transforms (conversion/qwen.py,
_LinearAttentionVReorderBase + Qwen3NextModel). Tensor NAMES are not hand-mapped:
the caller derives them by running the converter's own forward naming."""
import torch

NK, NV, DK, DV = 16, 32, 128, 128   # linear_num_key_heads, value_heads, key/value head dims
R = NV // NK


def unreorder(t: torch.Tensor, dim: int, hd: int) -> torch.Tensor:
    """Inverse of _reorder_v_heads: tiled [R, NK, hd] back to grouped [NK, R, hd]."""
    shape = list(t.shape)
    dim %= len(shape)
    new = shape[:dim] + [R, NK, hd] + shape[dim + 1:]
    t = t.reshape(new)
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(perm).contiguous().reshape(shape)


def invert(hf_name: str, t: torch.Tensor) -> torch.Tensor:
    """t: the GGUF tensor (dequantized, float32, HF orientation [out, in])."""
    t = t.float()
    # The converter adds 1 to every *filtered* name ending "norm.weight" except
    # linear_attn.norm; mtp.pre_fc_norm_{embedding,hidden} become "...enorm/hnorm.weight".
    if (hf_name.endswith("norm.weight") and not hf_name.endswith("linear_attn.norm.weight")) \
            or hf_name.endswith(("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")):
        t = t - 1
    if "linear_attn." not in hf_name:
        return t
    if hf_name.endswith(".A_log"):
        return unreorder(torch.log(-t).unsqueeze(-1), 0, 1).squeeze(-1)
    if hf_name.endswith(".dt_bias"):
        return unreorder(t.unsqueeze(-1), 0, 1).squeeze(-1)
    if ".conv1d" in hf_name:
        qk = DK * NK * 2
        t = torch.cat([t[:qk], unreorder(t[qk:], 0, DV)], dim=0)
        return t.unsqueeze(1)                      # [C, K] -> [C, 1, K]
    if ".in_proj_qkv." in hf_name:
        qk = DK * NK * 2
        return torch.cat([t[:qk], unreorder(t[qk:], 0, DV)], dim=0)
    if ".in_proj_z." in hf_name:
        return unreorder(t, 0, DV)
    if ".in_proj_a." in hf_name or ".in_proj_b." in hf_name:
        return unreorder(t, 0, 1)
    if ".out_proj." in hf_name:
        return unreorder(t, 1, DV)
    return t                                       # linear_attn.norm.weight
