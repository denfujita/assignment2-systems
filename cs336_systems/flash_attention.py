import torch
import einops
import math

class SpeedyAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # custom tile sizes
        b_q = 16
        b_k = 16 

        # split q, k, v into tiles
        tiled_q = einops.rearrange(Q, "b (t_q b_q) d -> b t_q b_q d", b_q=b_q)
        tiled_k = einops.rearrange(K, "b (t_k b_k) d -> b t_k b_k d", b_k=b_k)
        tiled_v = einops.rearrange(V, "b (t_k b_k) d -> b t_k b_k d", b_k=b_k)

        b, t_q, b_q, d = tiled_q.shape
        b, t_k, b_k, d = tiled_k.shape

        global_o_matrix = torch.zeros_like(tiled_q)
        global_l_matrix = torch.zeros((b, t_q, b_q), device=Q.device)

        for i in range(t_q):
            o_tile = torch.zeros((b, b_q, d), device=Q.device, dtype=Q.dtype)
            l = torch.zeros((b, b_q, ),  device=Q.device, dtype=Q.dtype)
            m = torch.full((b, b_q, ), float("-inf"), device=Q.device, dtype=Q.dtype)
            for j in range(t_k):
                s = einops.einsum(tiled_q[:, i], tiled_k[:, j], "b b_q d, b b_k d -> b b_q b_k")
                s /= math.sqrt(d)
                cur_row_max = torch.max(s, dim=2).values
                m_old = m
                m = torch.maximum(m_old, cur_row_max)
                p = torch.exp(s - m.unsqueeze(-1))
                l = torch.exp(m_old - m) * l + p.sum(dim=2)
                o_tile = torch.exp(m_old - m).unsqueeze(-1) * o_tile + p @ tiled_v[:, j]
            o_tile = o_tile / l.unsqueeze(-1)
            l_i = m + torch.log(l)
            global_o_matrix[:, i] = o_tile
            global_l_matrix[:, i] = l_i
        
        output_o = einops.rearrange(global_o_matrix, "b t_q b_q d -> b (t_q b_q) d")
        output_l = einops.rearrange(global_l_matrix, "b t_q b_q -> b (t_q b_q)")

        ctx.save_for_backward(output_o, output_l, tiled_q, tiled_k, tiled_v)
        return output_o

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
        