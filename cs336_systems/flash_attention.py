import torch


def reference_attention(q, k, v, is_causal=False):
    d = q.shape[-1]

    scores = torch.einsum("bqd,bkd->bqk", q, k)
    scores = scores / (d ** 0.5)

    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]

        q_positions = torch.arange(n_queries, device=q.device)
        k_positions = torch.arange(n_keys, device=k.device)

        causal_mask = q_positions[:, None] >= k_positions[None, :]
        scores = scores.masked_fill(~causal_mask, -1e6)

    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bqk,bkd->bqd", probabilities, v)

    return output

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        d = q.shape[-1]

        scores = torch.einsum("bqd,bkd->bqk", q, k)
        scores = scores / (d ** 0.5)

        if is_causal:
            n_queries = q.shape[-2]
            n_keys = k.shape[-2]

            q_positions = torch.arange(n_queries, device=q.device)
            k_positions = torch.arange(n_keys, device=k.device)

            causal_mask = q_positions[:, None] >= k_positions[None, :]
            scores = scores.masked_fill(~causal_mask, -1e6)

        lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.einsum("bqk,bkd->bqd", probabilities, v)

        ctx.save_for_backward(q, k, v, lse)
        ctx.is_causal = is_causal

        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, lse = ctx.saved_tensors
        is_causal = ctx.is_causal
        q_re = q.detach().requires_grad_(True)
        k_re = k.detach().requires_grad_(True)
        v_re = v.detach().requires_grad_(True)
        with torch.enable_grad():
            recomputed_output = reference_attention(
                q_re, k_re, v_re, is_causal
            )
        dq, dk, dv = torch.autograd.grad(
            recomputed_output,
            (q_re, k_re, v_re),
            grad_outputs=do,
        )
        return dq, dk, dv, None