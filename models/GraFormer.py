from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.ChebConv import ChebConv, _ResChebGC


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


gan_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4],
                          [0, 5], [5, 6], [6, 7], [7, 8],
                          [0, 9], [9, 10], [10, 11], [11, 12],
                          [0, 13], [13, 14], [14, 15], [15, 16],
                          [0, 17], [17, 18], [18, 19], [19, 20]], dtype=torch.long)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer[0](x,self.self_attn)
        return self.sublayer[1](x, self.feed_forward)

class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0.05, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # === [核心改进: 自动学习分组，不再使用手动列表] ===
        # 原本的手动列表 local_pooling_group 被废弃了
        # 我们让网络自己学出 8 个 Part 特征和 4 个 Region 特征
        self.num_cluster_1 = 8 
        self.num_cluster_2 = 4

        # 定义两个线性层，用来计算每个节点属于哪个组的“权重”
        # 输入维度 dim -> 输出维度 num_cluster
        self.assign_1 = nn.Linear(dim, self.num_cluster_1)
        self.assign_2 = nn.Linear(dim, self.num_cluster_2)
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 这里的 Dropout 已经默认设为你需要的 0.05
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: [Batch, 17, Dim]
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # === [核心改进: 动态聚合特征] ===
        
        # 1. 原始节点 (17个)
        pool_0 = x  # (B, 17, C)

        # 2. 学习聚合出 8 个特征 (代替原来的 pool_1)
        # 计算分配权重，并归一化
        assign_score_1 = F.softmax(self.assign_1(x), dim=1) 
        # 加权求和: (B, 8, 17) x (B, 17, C) -> (B, 8, C)
        pool_1 = torch.matmul(assign_score_1.transpose(1, 2), x)

        # 3. 学习聚合出 4 个特征 (代替原来的 pool_2)
        assign_score_2 = F.softmax(self.assign_2(x), dim=1)
        pool_2 = torch.matmul(assign_score_2.transpose(1, 2), x)

        # 拼接: 17 + 8 + 4 = 29 个节点
        pools_concat = torch.cat([pool_0, pool_1, pool_2], dim=1)
        
        # 归一化
        pools_concat = self.norm(pools_concat)

        # === 后面的 Attention 计算逻辑不变 ===
        
        # 计算 K, V (基于拼接后的 29 个节点)
        kv = self.kv(pools_concat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Attention: 原始17个节点去查询这29个节点
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True]]])


class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, n_pts=21):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1


class GraFormer(nn.Module):
    def __init__(self, adj, hid_dim=128, coords_dim=(2, 3), num_layers=4,
                 n_head=4,  dropout=0.1, n_pts=21):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        self.adj = adj

        _gconv_input = ChebConv(in_c=coords_dim[0], out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = hid_dim
        c = copy.deepcopy
        attn = PoolingAttention(num_heads=n_head, dim = dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                                hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=3, K=2)

    def forward(self, x):
        out = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            out = self.atten_layers[i](out)
            out = self.gconv_layers[i](out)
        out = self.gconv_output(out, self.adj)
        return out


if __name__ == '__main__':
    adj = adj_mx_from_edges(num_pts=21, edges=gan_edges, sparse=False)
    model = GraFormer(adj=adj, hid_dim=128)
    x = torch.zeros((1, 21, 2))
    print(model(x, src_mask))

