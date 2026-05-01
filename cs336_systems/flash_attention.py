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
        output_o, output_l, tiled_q, tiled_k, tiled_v = ctx.saved_tensors

        b_q = tiled_q.shape[2]
        b_k = tiled_k.shape[2]

        b, t_q, b_q, d = tiled_q.shape
        _, t_k, b_k, _ = tiled_k.shape

        # tile dO 
        tiled_dO = einops.rearrange(
            grad_output,
            "b (t_q b_q) d -> b t_q b_q d",
            b_q=b_q,
        )

        # D = rowsum(O * dO) ...
        # Shape: [b, t_q, b_q]
        tiled_D = torch.sum(
            output_o.reshape(b, t_q, b_q, d) * tiled_dO,
            dim=-1,
        )

        dQ = torch.zeros_like(tiled_q)
        dK = torch.zeros_like(tiled_k)
        dV = torch.zeros_like(tiled_v)

        scale = 1.0 / math.sqrt(d)

        # compute dK and dV
        for j in range(t_k):
            k_j = tiled_k[:, j]  
            v_j = tiled_v[:, j]  

            dK_j = torch.zeros_like(k_j)
            dV_j = torch.zeros_like(v_j)

            for i in range(t_q):
                q_i = tiled_q[:, i]       
                dO_i = tiled_dO[:, i]     
                L_i = output_l.reshape(b, t_q, b_q)[:, i]  
                D_i = tiled_D[:, i]       

                # S = Q K^T / sqrt(d)
                S = einops.einsum(q_i, k_j, "b b_q d, b b_k d -> b b_q b_k") * scale

                # P = exp(S - L)
                P = torch.exp(S - L_i.unsqueeze(-1))  # [b, b_q, b_k]

                # dV += P^T dO
                dV_j += einops.einsum(P, dO_i,"b b_q b_k, b b_q d -> b b_k d")

                # dP = dO V^T
                dP = einops.einsum(dO_i, v_j,"b b_q d, b b_k d -> b b_q b_k")

                # dS = P ⊙ (dP - D)
                dS = P * (dP - D_i.unsqueeze(-1))

                # dK += dS^T Q / sqrt(d)
                dK_j += einops.einsum(dS, q_i, "b b_q b_k, b b_q d -> b b_k d") * scale

            dK[:, j] = dK_j
            dV[:, j] = dV_j

        # Second pass: compute dQ
        for i in range(t_q):
            q_i = tiled_q[:, i]
            dO_i = tiled_dO[:, i]
            L_i = output_l.reshape(b, t_q, b_q)[:, i]
            D_i = tiled_D[:, i]

            dQ_i = torch.zeros_like(q_i)

            for j in range(t_k):
                k_j = tiled_k[:, j]
                v_j = tiled_v[:, j]

                S = einops.einsum(q_i, k_j,"b b_q d, b b_k d -> b b_q b_k") * scale

                P = torch.exp(S - L_i.unsqueeze(-1))

                dP = einops.einsum(dO_i, v_j, "b b_q d, b b_k d -> b b_q b_k")

                dS = P * (dP - D_i.unsqueeze(-1))

                # dQ += dS K / sqrt(d)
                dQ_i += einops.einsum(dS, k_j,"b b_q b_k, b b_k d -> b b_q d") * scale

            dQ[:, i] = dQ_i

        dQ = einops.rearrange(dQ, "b t_q b_q d -> b (t_q b_q) d")
        dK = einops.rearrange(dK, "b t_k b_k d -> b (t_k b_k) d")
        dV = einops.rearrange(dV, "b t_k b_k d -> b (t_k b_k) d")

        return dQ, dK, dV, None