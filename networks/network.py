import torch.nn as nn
from torch.nn.functional import normalize
import torch
from torch.nn import Parameter
from typing import Optional

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class DEC(nn.Module):
    def __init__(self, class_num, hidden_dimension, alpha: float = 1.0, cluster_centers: Optional[torch.Tensor] = None):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param class_num: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.class_num = class_num
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.class_num, self.hidden_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)


    def forward(self, batch):
        """
        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device, batch_size):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.decs = []

        # 增加深度聚类模块
        self.alpha = 1.0
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
            self.decs.append(DEC(class_num, high_feature_dim, self.alpha).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.decs = nn.ModuleList(self.decs)
        
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(high_feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view

        self.lowrank = 20
        self.lam = 1.0
    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            u=z

            # u, _, _ = torch.svd(z)
            # u = u[:, :self.lowrank]

            W = torch.abs(u.matmul(torch.transpose(u,dim0=1,dim1=0)))
            W.squeeze().fill_diagonal_(0)
            isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(W, dim=-1,keepdim=True))
            W = W * isqrt_diag * torch.transpose(isqrt_diag,dim0=1,dim1=0)

            W1 = torch.ones_like(W)
            W1.squeeze().fill_diagonal_(0)
            isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(W1, dim=-1,keepdim=True))
            W1 = W1 * isqrt_diag * torch.transpose(isqrt_diag,dim0=1,dim1=0)

            adj_normalized = normalize(W1 - self.lam * W, dim=1)

            h = normalize(self.feature_contrastive_module(torch.mm(adj_normalized, z)), dim=1)
            q = self.label_contrastive_module(h)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.feature_contrastive_module(z)
            q = self.label_contrastive_module(h)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds
