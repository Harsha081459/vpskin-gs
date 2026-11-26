"""Drop-in replacement for `knn_cuda.KNN` (transpose_mode=True) in pure torch.

knn_cuda JIT-compiles on import and is fragile on new torch/CUDA images (Kaggle).
GauHuman only uses KNN(k=1) between SMPL vertices (6890) and Gaussian centres,
so chunked cdist is fast enough.
"""
import torch


class KNN:
    def __init__(self, k: int = 1, transpose_mode: bool = True, chunk: int = 65536):
        assert transpose_mode, "only transpose_mode=True (B, N, 3) inputs are supported"
        self.k, self.chunk = k, chunk

    @torch.no_grad()
    def __call__(self, ref: torch.Tensor, query: torch.Tensor):
        """ref (B, Nr, D), query (B, Nq, D) -> dist (B, Nq, k), idx (B, Nq, k) long."""
        dists, idxs = [], []
        for s in range(0, query.shape[1], self.chunk):
            d = torch.cdist(query[:, s:s + self.chunk].float(), ref.float())
            v, i = d.topk(self.k, dim=-1, largest=False)
            dists.append(v)
            idxs.append(i)
        return torch.cat(dists, 1), torch.cat(idxs, 1)
