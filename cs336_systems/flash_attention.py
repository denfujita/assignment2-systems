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

        global_o_matrix = torch.zeros((b, t_q, b_q, d), device=Q.device, dtype=torch.float32)
        global_l_matrix = torch.zeros((b, t_q, b_q), device=Q.device, dtype=torch.float32)

        for i in range(t_q):
            o_tile = torch.zeros((b, b_q, d), device=Q.device, dtype=torch.float32)
            l = torch.zeros((b, b_q), device=Q.device, dtype=torch.float32)
            m = torch.full((b, b_q), float("-inf"), device=Q.device, dtype=torch.float32)
            q_i = tiled_q[:, i].float()
            
            for j in range(t_k):
                k_j = tiled_k[:, j].float()
                v_j = tiled_v[:, j].float()

                s = einops.einsum(q_i, k_j, "b b_q d, b b_k d -> b b_q b_k")
                s /= math.sqrt(d)

                if is_causal:
                    q_pos = i * b_q + torch.arange(b_q, device=Q.device)
                    k_pos = j * b_k + torch.arange(b_k, device=Q.device)
                    mask = k_pos[None, :] <= q_pos[:, None]
                    s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))

                cur_row_max = torch.max(s, dim=2).values
                m_old = m
                m = torch.maximum(m_old, cur_row_max)
                p = torch.exp(s - m.unsqueeze(-1))
                l = torch.exp(m_old - m) * l + p.sum(dim=2)
                o_tile = torch.exp(m_old - m).unsqueeze(-1) * o_tile + p @ v_j
            o_tile = o_tile / l.unsqueeze(-1)
            l_i = m + torch.log(l)
            global_o_matrix[:, i] = o_tile
            global_l_matrix[:, i] = l_i
        
        output_o_fp32 = einops.rearrange(global_o_matrix, "b t_q b_q d -> b (t_q b_q) d")
        output_o = output_o_fp32.to(Q.dtype)
        output_l = einops.rearrange(global_l_matrix, "b t_q b_q -> b (t_q b_q)")

        ctx.save_for_backward(output_o_fp32, output_l, tiled_q, tiled_k, tiled_v)
        ctx.is_causal = is_causal
        return output_o

    @staticmethod
    def backward(ctx, grad_output):
        output_o, output_l, tiled_q, tiled_k, tiled_v = ctx.saved_tensors
        q_dtype = tiled_q.dtype
        k_dtype = tiled_k.dtype
        v_dtype = tiled_v.dtype
        is_causal = ctx.is_causal

        b, t_q, b_q, d = tiled_q.shape
        _, t_k, b_k, _ = tiled_k.shape

        tiled_dO = einops.rearrange(grad_output, "b (t_q b_q) d -> b t_q b_q d",b_q=b_q ).float()

        tiled_L = einops.rearrange(output_l, "b (t_q b_q) -> b t_q b_q", b_q=b_q)

        tiled_O = einops.rearrange(output_o, "b (t_q b_q) d -> b t_q b_q d", b_q=b_q,).float()

        tiled_D = torch.sum(tiled_O * tiled_dO, dim=-1)

        dQ = torch.zeros_like(tiled_q, dtype=torch.float32)
        dK = torch.zeros_like(tiled_k, dtype=torch.float32)
        dV = torch.zeros_like(tiled_v, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d)

        # compute dK and dV
        for j in range(t_k):
            dK_j = torch.zeros((b, b_k, d), device=tiled_k.device, dtype=torch.float32)
            dV_j = torch.zeros((b, b_k, d), device=tiled_v.device, dtype=torch.float32)

            k_j = tiled_k[:, j].float()
            v_j = tiled_v[:, j].float()

            for i in range(t_q):
                q_i = tiled_q[:, i].float()
                dO_i = tiled_dO[:, i]
                L_i = tiled_L[:, i]
                D_i = tiled_D[:, i]

                S = einops.einsum(q_i, k_j, "b b_q d, b b_k d -> b b_q b_k") * scale

                if is_causal:
                    q_pos = i * b_q + torch.arange(b_q, device=S.device)
                    k_pos = j * b_k + torch.arange(b_k, device=S.device)
                    mask = k_pos[None, :] <= q_pos[:, None]
                    S = S.masked_fill(~mask.unsqueeze(0), float("-inf"))

                P = torch.exp(S - L_i.unsqueeze(-1))

                dV_j += einops.einsum(P, dO_i, "b b_q b_k, b b_q d -> b b_k d")

                dP = einops.einsum(dO_i, v_j, "b b_q d, b b_k d -> b b_q b_k")

                dS = P * (dP - D_i.unsqueeze(-1))

                dK_j += einops.einsum(dS, q_i, "b b_q b_k, b b_q d -> b b_k d") * scale

            dK[:, j] = dK_j
            dV[:, j] = dV_j

        # compute dQ
        for i in range(t_q):
            q_i = tiled_q[:, i].float()
            dO_i = tiled_dO[:, i]
            L_i = tiled_L[:, i]
            D_i = tiled_D[:, i]

            dQ_i = torch.zeros((b, b_q, d), device=tiled_q.device, dtype=torch.float32)

            for j in range(t_k):
                k_j = tiled_k[:, j].float()
                v_j = tiled_v[:, j].float()

                S = einops.einsum(q_i, k_j, "b b_q d, b b_k d -> b b_q b_k") * scale

                if is_causal:
                    q_pos = i * b_q + torch.arange(b_q, device=S.device)
                    k_pos = j * b_k + torch.arange(b_k, device=S.device)
                    mask = k_pos[None, :] <= q_pos[:, None]
                    S = S.masked_fill(~mask.unsqueeze(0), float("-inf"))

                P = torch.exp(S - L_i.unsqueeze(-1))

                dP = einops.einsum(dO_i, v_j, "b b_q d, b b_k d -> b b_q b_k")

                dS = P * (dP - D_i.unsqueeze(-1))

                dQ_i += einops.einsum(dS, k_j, "b b_q b_k, b b_k d -> b b_q d") * scale

            dQ[:, i] = dQ_i

        dQ = einops.rearrange(dQ, "b t_q b_q d -> b (t_q b_q) d").to(q_dtype)
        dK = einops.rearrange(dK, "b t_k b_k d -> b (t_k b_k) d").to(k_dtype)
        dV = einops.rearrange(dV, "b t_k b_k d -> b (t_k b_k) d").to(v_dtype)

        return dQ, dK, dV, None