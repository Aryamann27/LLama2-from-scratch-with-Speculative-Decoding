import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Transformer
from typing import Optional

def verify_with_target(
    model: Transformer,
    tokens: torch.Tensor,   # (B, K+1) — [last_prefix, d1, d2, ..., dK]
    start_pos_first: int,   # position of the first token in the full sequence (e.g. len(prefix)-1)
) -> torch.Tensor:
    """Returns logits (B, K+1, vocab_size) for the next token at each position."""
    B, L = tokens.shape
    logits_list = []

    for i in range(L):
        # One token per batch at position (start_pos_first + i)
        token_i = tokens[:, i : i + 1]   # (B, 1)
        pos_i = start_pos_first + i
        logits_i = model(token_i, pos_i)  # (B, 1, vocab_size) — full model, not draft
        logits_list.append(logits_i)

    # Stack: list of (B, 1, V) -> (B, L, V)
    return torch.cat(logits_list, dim=1)

def accept_tokens(
    draft_tokens: torch.Tensor,
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,) -> list:

    B, K, V = draft_tokens.size(0), draft_tokens.size(1), logits.size(-1)
    device = logits.device
    accepted = [[] for _ in range(B)]
    done = [False] * B

    for i in range(K+1):
        logits_i = logits[:, i, :]

        if temperature <=0:
            next_tokens = logits_i.argmax(dim=-1)
        else:
            probs = F.softmax(logits_i / temperature, dim=-1)
            if top_k > 0:
                v, _ = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                probs = (probs >= v[:, -1:]).float() * probs
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1) # (B,)
        
        for b in range(B):
            if done[b]:
                continue
            accepted[b].append(next_tokens[b].item())
            if i < K and next_tokens[b].item() != draft_tokens[b, i].item():
                done[b] = True   # target disagreed; stop accepting for this sequence
            if i == K:
                for b in range(B):
                    done[b] = True
    
    return accepted


def _sample_top_p_from_logits(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """logits (B, V) -> sample (B, 1)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, 1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def speculative_step(
    model: Transformer,
    tokens: torch.Tensor,       # (B, total_len); valid prefix is tokens[:, :cur_pos]
    cur_pos: int,               # length of current prefix (next token to fill is at cur_pos)
    draft_k: int,
    layer_indices: list,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> list:
    """
    Run draft (subset of layers) K times, then verify with full model.
    Returns list of length B: each element is a list of accepted token IDs (1 to K+1).
    """
    B, total_len = tokens.shape
    device = tokens.device
    last_token = tokens[:, cur_pos - 1 : cur_pos]   # (B, 1) — last token of current prefix

    # --- Draft: K steps with draft model ---
    draft_tokens_list = []
    current_token = last_token
    for k in range(draft_k):
        start_pos = cur_pos - 1 + k   # position of current token in sequence
        logits_d = model.forward_draft(current_token, start_pos, layer_indices=layer_indices)
        logits_d = logits_d[:, -1, :]   # (B, V)
        next_tok = _sample_top_p_from_logits(logits_d, temperature, top_p)   # (B, 1)
        draft_tokens_list.append(next_tok)
        current_token = next_tok

    draft_tokens = torch.cat(draft_tokens_list, dim=1)   # (B, K)

    # --- Verify: full model on [last_token, d1, ..., dK] ---
    tokens_to_verify = torch.cat([last_token, draft_tokens], dim=1)   # (B, K+1)
    start_pos_first = cur_pos - 1   # position of first token in this segment
    logits = verify_with_target(model, tokens_to_verify, start_pos_first)   # (B, K+1, V)

    # --- Accept (uses temperature/top_p via accept_tokens) ---
    accepted = accept_tokens(draft_tokens, logits, temperature=temperature, top_k=0)
    return accepted