import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from gather_layer import GatherLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def target_distribution(batch):
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

    
class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, high_feature_dim,  device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.dec_loss = nn.KLDivLoss(reduction="batchmean")
        self.predictor = nn.Sequential(
            nn.Linear(high_feature_dim, class_num),
            nn.Softmax(dim=1)
        )

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask



    def get_samples_mask(self, N, sim):
        positive_mask = torch.zeros((N, N))
        negative_mask = torch.ones((N, N))
        negative_mask = negative_mask.fill_diagonal_(0)
        K = 1
        topK = 2*K
        for i in range(self.batch_size):

            negative_mask[i, self.batch_size + i] = 0
            positive_mask[i, self.batch_size + i] = 1


            negative_mask[self.batch_size + i, i] = 0
            positive_mask[self.batch_size + i, i] = 1


            view_i = sim[i, self.batch_size:]
            view_j = sim[self.batch_size+i, :self.batch_size]
            cur_sample = torch.cat((view_i, view_j), dim=0)


            _, max_value_index = cur_sample.topk(topK+2)
            cnt = 0
            index = 0
            while(cnt < topK and index < topK + 2):
                if (max_value_index[index] != self.batch_size + i and max_value_index[index] != i):
                    if (max_value_index[index] >= self.batch_size):
                            positive_mask[self.batch_size + i, max_value_index[index] - self.batch_size] = 1
                            negative_mask[self.batch_size + i, max_value_index[index] - self.batch_size] = 0
                    else:
                        positive_mask[i, max_value_index[index] + self.batch_size] = 1
                        negative_mask[i, max_value_index[index] + self.batch_size] = 0

                    cnt += 1
                index += 1
        positive_mask = positive_mask.bool()
        negative_mask = negative_mask.bool()

        return (positive_mask, negative_mask)




    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) 
        loss /= N
        return loss


    def forward_neighbor_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f


        (positive_mask, negative_mask) = self.get_samples_mask(N, sim)
        positive_samples = sim[positive_mask].reshape(N, -1)
        negative_samples = sim[negative_mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss




    def forward_label(self, q_i, q_j):
        K = self.class_num
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy 

    def forward_dec(self, h):
        target = target_distribution(h).detach()
        loss = self.dec_loss(h.log(), target) / h.shape[0]
        return loss

    def forward_cluster_center(self, c_i, c_j):
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)


        sim = torch.matmul(c, c.T) / 1.0
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss



    def forward_sample_cluster_center(self, h_i, c_j):
        N = 2 * self.class_num
        print('h_i:', h_i.shape)
        print('c_j:', c_j.shape)
        c = torch.cat((h_i, c_j), dim=0)


        sim = torch.matmul(c, c.T) / 1.0
        print('sim:', sim.shape)
        sim_i_j = torch.diag(sim, self.class_num+self.batch_size)
        sim_j_i = torch.diag(sim, -(self.class_num+self.batch_size))
        M = 2*(self.class_num+self.batch_size)
        print(torch.cat((sim_i_j, sim_j_i), dim=0).shape)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(M, 1)
        mask = self.mask_correlated_samples(M)
        negative_samples = sim[mask].reshape(M, -1)

        labels = torch.zeros(M).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= M
        return loss


    def compute_cluster_loss(self, q_centers, k_centers):


        self.num_cluster = self.class_num
        d_q = q_centers.mm(q_centers.T) / 1.0   #1.0
        d_k = (q_centers * k_centers).sum(dim=1) / 1.0
        d_q = d_q.float()
        d_q[torch.arange(self.num_cluster), torch.arange(self.num_cluster)] = d_k
        mask = torch.zeros((self.num_cluster, self.num_cluster), dtype=torch.bool, device=d_q.device)
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.num_cluster, self.num_cluster))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.num_cluster - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.num_cluster, 1), neg], dim=1), dim=1) 

        loss = loss.sum() / (self.num_cluster)
        return 1.0 * loss


    def compute_centers(self, x, psedo_labels):
        self.num_cluster = self.class_num
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.num_cluster, n_samples).to(x)  # L, N
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=2, dim=1)  # l1 normalization
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers



    def forward_cluster_result(self, p_i, p_j):
        N = 2 * self.class_num
        p = torch.cat((p_i, p_j), dim=0)

        sim = torch.matmul(p, p.T) / 1.0
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss








