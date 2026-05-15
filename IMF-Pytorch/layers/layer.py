import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA = torch.cuda.is_available()


class ConvKBLayer(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super(ConvKBLayer, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels, (1, input_seq_len))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear(input_dim * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class MutanLayer(nn.Module):
    def __init__(self, dim, multi):
        super(MutanLayer, self).__init__()

        self.dim = dim
        self.multi = multi

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(torch.mul(torch.mul(x_modal1, x_modal2), x_modal3))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)
        return x_mm


class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()
        
        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))
        
        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)
       
        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)
        return x


class ConvELayer(nn.Module):
    def __init__(self, dim, out_channels, kernel_size, k_h, k_w):
        super(ConvELayer, self).__init__()

        self.input_drop = nn.Dropout(0.2)
        self.conv_drop = nn.Dropout2d(0.2)
        self.hidden_drop = nn.Dropout(0.2)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm1d(dim)

        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                                    stride=1, padding=0, bias=True)
        assert k_h * k_w == dim
        flat_sz_h = int(2*k_w) - kernel_size + 1
        flat_sz_w = k_h - kernel_size + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = nn.Linear(self.flat_sz, dim, bias=True)

    def forward(self, conv_input):
        x = self.bn0(conv_input)
        x = self.input_drop(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            if(CUDA):
                edge_sources = edge_sources.to('cuda:0')
            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(out_features, 2 * in_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge):
        N = input.size()[0]

        edge_h = torch.cat((input[edge[0, :], :], input[edge[1, :], :]), dim=1).t()
        # edge_h: (2*in_dim) x E

        edge_m = self.W.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_m).squeeze())).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(edge, edge_w, N, edge_w.shape[0], self.out_features)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(nfeat, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nheads * nhid,
                                             nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, entity_embeddings, relation_embeddings, edge_list):
        x = entity_embeddings
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x_rel = relation_embeddings.mm(self.W)
        x = F.elu(self.out_att(x, edge_list))
        return x, x_rel


class SpDHRGATLayer(nn.Module):
    """
    动态层次化关系感知图注意力网络（局部关系感知聚合层）
    对应文档中: 局部关系感知聚合 (Local Relation-aware Aggregation)
    """

    def __init__(self, num_nodes, in_features, out_features, dropout, alpha, concat=True):
        super(SpDHRGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat

        # 关系动态旋转参数：由两层MLP生成旋转角 theta_r
        self.theta_mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, in_features)
        )
        self.c_r = nn.Parameter(torch.zeros(1, in_features))

        # 共享投影矩阵 W 和 关系特定投影矩阵 W_r
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.W_r = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W_r.data, gain=1.414)

        # 注意力计算参数 a (由于拼接了三部分，维度是 3 * out_features)
        self.a = nn.Parameter(torch.zeros(size=(1, 3 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, relation_embeds, edge_list, edge_type):
        N = input.size()[0]

        # 提取头尾实体嵌入
        h_i = input[edge_list[0, :], :]
        h_j = input[edge_list[1, :], :]

        # 获取对应边的关系嵌入
        r_ij = relation_embeds[edge_type]

        # 动态旋转关系编码 (公式 3-5): 模拟复数空间旋转取实部映射
        theta_r = self.theta_mlp(r_ij)
        h_i_r = h_i * torch.cos(theta_r) + self.c_r

        # 映射到统一空间
        W_h_i = torch.mm(h_i, self.W)
        W_r_h_i_r = torch.mm(h_i_r, self.W_r)
        W_h_j = torch.mm(h_j, self.W)

        # 注意力输入向量拼接: x_ij = [Whi || Wr hir || Whj] (公式 3-6)
        x_ij = torch.cat([W_h_i, W_r_h_i_r, W_h_j], dim=-1)

        # 计算注意力分数 (公式 3-7)
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(x_ij.t()).squeeze())).unsqueeze(1)

        # 稀疏矩阵求和作为归一化分母
        e_rowsum = self.special_spmm_final(edge_list, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        edge_e = edge_e.squeeze(1)
        edge_e = self.dropout(edge_e)

        # 邻居信息加权 (公式 3-8)
        edge_w = (edge_e.unsqueeze(1) * W_h_j)

        # 加权聚合
        h_local = self.special_spmm_final(edge_list, edge_w, N, edge_w.shape[0], self.out_features)
        h_local = h_local.div(e_rowsum)

        if self.concat:
            return F.elu(h_local)
        else:
            return h_local


class GlobalPropagation(nn.Module):
    """全局结构传播 (Global Structure Propagation)"""

    def __init__(self, K=3):
        super(GlobalPropagation, self).__init__()
        self.K = K

    def forward(self, h_local, edge_list, num_nodes):
        device = h_local.device
        # 添加自环
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        edge_list_with_loops = torch.cat([edge_list, self_loops], dim=1)

        # 计算度矩阵
        deg = torch.bincount(edge_list_with_loops[0, :], minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # S = D^(-1/2) A D^(-1/2) (公式 3-9)
        row, col = edge_list_with_loops
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 构造稀疏传播矩阵
        adj_sparse = torch.sparse_coo_tensor(edge_list_with_loops, edge_weight, torch.Size([num_nodes, num_nodes]))

        # 迭代多跳传播 (公式 3-10, 3-11)
        H = h_local
        for _ in range(self.K):
            H = torch.sparse.mm(adj_sparse, H)

        return H


class SpDHRGAT(nn.Module):
    """层次化融合机制层: Fuse(局部, 全局)"""

    def __init__(self, num_nodes, nfeat, nhid, dropout, alpha, nheads):
        super(SpDHRGAT, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)

        # 多头局部注意力
        self.attentions = nn.ModuleList([
            SpDHRGATLayer(num_nodes, nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])

        self.W = nn.Parameter(torch.zeros(size=(nfeat, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 全局传播模块 K=3
        self.global_prop = GlobalPropagation(K=3)

        # 层次化融合映射层
        self.fuse_layer = nn.Linear(nheads * nhid * 2, nheads * nhid)

    def forward(self, entity_embeddings, relation_embeddings, edge_list, edge_type):
        x = entity_embeddings

        # 局部图注意力输出
        h_local = torch.cat([att(x, relation_embeddings, edge_list, edge_type) for att in self.attentions], dim=1)
        h_local = self.dropout_layer(h_local)

        # 全局多跳传播输出
        h_global = self.global_prop(h_local, edge_list, x.size(0))

        # 层次化融合 (公式 3-12)
        h_fused = F.elu(self.fuse_layer(torch.cat([h_local, h_global], dim=-1)))

        x_rel = relation_embeddings.mm(self.W)
        return h_fused, x_rel

class GSMSF_Gate(nn.Module):
    """
    GS-MSF Gate (Graph-Structure guided Multi-Stage Filtering Gate)
    用于多模态知识图谱补全的前置图像过滤
    """

    def __init__(self, lambda1=0.3, lambda2=0.4, lambda3=0.3, k=5):
        super(GSMSF_Gate, self).__init__()
        # λ1, λ2, λ3 为各相似度分量的权重系数
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        # k 为经验参考值（默认为 5）
        self.k = k

    def forward(self, phash_sim, clip_sim, ssim_sim, num_metapaths, shortest_path_dist, candidate_num):
        """
        第二层级：图结构引导的多粒度图像精筛
        第三层级：自适应阈值选择
        """
        # 计算多粒度图像相似度 sim(v_i, v_j) (公式 4-2)
        sim_vi_vj = self.lambda1 * phash_sim + self.lambda2 * clip_sim + self.lambda3 * ssim_sim

        # 计算图结构关联权重 w_ij (与元路径数量成正比，与最短路径距离成反比) (公式 4-3)
        # 加上 1e-5 防止除零错误
        w_ij = num_metapaths / (shortest_path_dist + 1e-5)

        # 综合相似度得分通过 Softmax 归一化形式计算 (公式 4-4)
        alpha_ij = F.softmax(sim_vi_vj * w_ij, dim=-1)

        # 计算自适应阈值 tau_e (公式 4-5)
        mu = torch.mean(sim_vi_vj)
        sigma = torch.std(sim_vi_vj)
        # 根据当前实体候选图像数量 |I_e| 定义自适应阈值
        tau_e = mu + sigma * torch.tanh(torch.tensor(candidate_num - self.k, dtype=torch.float32))

        return alpha_ij, tau_e

