import torch
import triton # type: ignore[import-not-found]
import triton.language as tl # type: ignore[import-not-found]

class SpeedyTritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V):
        

@triton.jit # type: ignore[import-not-found]
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    l_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE, ),
    )
    # running qk^T max
    m_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - float("inf")
    # curr softmax denominator
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)

    # result tile to send back to HBM 
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    #assignment guarantees whole tiles, no need for mask
    q_tile = tl.load(Q_block_ptr)

    tile_k_arange = tl.arange(0, K_TILE_SIZE)

    o_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    for kv_idx in tl.range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        kv_token_idx = kv_idx *  K_TILE_SIZE

        k_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(kv_token_idx, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        kt_tile = tl.trans(tl.load(k_ptr))

        v_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(kv_token_idx, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        v_tile = tl.load(v_ptr)

        qk = tl.dot(
            q_tile * scale,
            kt_tile,
            out_dtype=tl.float32
        )

        # masking 
        # kv_indices = kv_token_idx + tile_k_arange
        # mask = q_lens_mask & (
        #   kv_indices[None, :] < N_KEYS
        # )
        # qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))

        # max over seq len
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        # e^(x2 - m(x))
        p = tl.math.exp(qk - m_ij[:, None])

        # cur rolling sum
        l_ij = tl.sum(p, 1)

        l_i = l_i * tl.math.exp(m_i - m_ij) + l_ij

        acc = acc * tl.math.exp(m_i - m_ij)[:, None]

        acc += tl.dot(p, v_tile)

        m_i = m_ij
    
    acc = acc / l_i[:, None]
    tl.store(l_ptr, m_i + tl.math.log(l_i))
    tl.store(o_ptr, acc)