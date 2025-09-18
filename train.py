import torch
from network import Network
from metric import valid

import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os

import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


start_time = time.time()

# BDGP
# CCV
# Fashion

Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0005)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.

if args.dataset == "BDGP":
    args.batch_size = 64
    args.mse_epochs = 20
    args.con_epochs = 100
    seed = 10

if args.dataset == "CCV":
    args.batch_size = 256
    args.mse_epochs = 200
    args.con_epochs = 100
    seed = 3
    
if args.dataset == "Fashion":
    args.mse_epochs = 50
    args.con_epochs = 100
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            # xs[v] = 
            xs[v] = xs[v].to(device)
            
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print((tot_loss / len(data_loader)))


def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    cross_entropy = torch.nn.CrossEntropyLoss()
    predicted = [None] * view
    for v in range(view):
        predicted[v] = []

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        alpha = 1 #h
        beta = 0  #q

        gamma = 0.0001 #c

        zeta = 0.0001  #self-supervised

        for v in range(view):
            xs[v] = xs[v].to(device)
            
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mes(xs[v], xrs[v]))

            cluster_center_v = criterion.compute_centers(hs[v], qs[v].argmax(1))            
            cluster_results_v = model.decs[v](hs[v])

            if torch.cuda.is_available():
                    cluster_center_v = cluster_center_v.cuda(non_blocking=True)
            with torch.no_grad():
                model.state_dict()["decs.{}.cluster_centers".format(v)].copy_(cluster_center_v)

            for w in range(v+1, view):

                loss_list.append(alpha*criterion.forward_neighbor_feature(hs[v], hs[w]))   
                loss_list.append(beta*criterion.forward_label(qs[v], qs[w]))     
                cluster_center_w = criterion.compute_centers(hs[w], qs[w].argmax(1))                    
                if torch.cuda.is_available():
                    cluster_center_w = cluster_center_w.cuda(non_blocking=True)
                with torch.no_grad():
                    model.state_dict()["decs.{}.cluster_centers".format(w)].copy_(cluster_center_w)

                loss_list.append(gamma*criterion.compute_cluster_loss(cluster_center_v, cluster_center_w)) 

            loss_list.append(zeta * kl_loss(qs[v].log(), cluster_results_v))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print((tot_loss / len(data_loader)))

accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, args.high_feature_dim, device).to(device)

    epoch = 1
    stopping_delta=0.000001

    print('begin the pretrain the auto-encoder')
    while epoch <= args.mse_epochs:  
        pretrain(epoch)
        if epoch == args.mse_epochs:
            nmi, ari, acc, pur, total_pred, pred_vectors, high_level_vectors, labels_vector, low_level_vectors, Centers, prototype_feature = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
            
        epoch += 1



    while epoch <= args.mse_epochs + args.con_epochs:
        mse_con = epoch-args.mse_epochs
        contrastive_train(epoch)
        if epoch == args.mse_epochs + args.con_epochs:
            nmi, ari, acc, pur, total_pred, pred_vectors, high_level_vectors, labels_vector, low_level_vectors, Centers, prototype_feature = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
        epoch += 1

    
end_time = time.time()
execution_time = end_time - start_time
print("Total time", execution_time)

