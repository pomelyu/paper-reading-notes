from torch import nn, tensor
from torch.nn.functional import grid_sample


class DeformAttn(nn.Module):
    def __init__(self, C=256, K=4):
        self.K = K
        self.offsets = nn.Linear(C, K * 2)   # 去哪取樣
        self.weights = nn.Linear(C, K)       # 每個取樣點多重要
        self.value   = nn.Linear(C, C)
        self.out     = nn.Linear(C, C)

    def forward(self, query, ref, feat):
        # query (N, Lq, C)      物件 query
        # ref   (N, Lq, 2)      每個 query 的參考點，normalized [0,1]
        # feat  (N, C, H, W)    backbone 出來的 feature map
        N, Lq, C = query.shape
        H, W = feat.shape[-2:]

        v = self.value(feat)                                   # (N, C, H, W)

        # 1. query 自己吐出 K 個偏移量和 K 個權重（完全沒有用到 feat）
        d = self.offsets(query).view(N, Lq, self.K, 2)          # 單位是 pixel
        a = self.weights(query).softmax(-1)                     # (N, Lq, K)，跨 K 做 softmax

        # 2. 參考點 + 偏移 = 實際取樣座標
        loc = ref[:, :, None, :] + d / tensor([W, H])           # (N, Lq, K, 2)

        # 3. 在這 K 個連續座標上做 bilinear 取值
        s = grid_sample(v, 2 * loc - 1)                         # (N, C, Lq, K)

        # 4. 加權求和
        o = (s * a.permute(0, 2, 1)[:, None]).sum(-1)           # (N, C, Lq)
        return self.out(o.transpose(1, 2))                      # (N, Lq, C)
