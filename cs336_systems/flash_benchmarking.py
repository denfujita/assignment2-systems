import torch

def _make_attn_inputs(device=None):
    torch.random.manual_seed(0)
    batch_size = 4
    n_queries = 128
    n_keys = 128
    D = 64
    q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    do = torch.randn(batch_size, n_queries, D, device=device)

    return q, k, v, do