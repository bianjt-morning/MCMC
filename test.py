import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data
import scipy.io as sio

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--high_feature_dim", default=256)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
nmi, ari, acc, pur, total_pred, pred_vectors, high_level_vectors, labels_vector, low_level_vectors, Centers, prototype_feature = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
# sio.savemat('CIFAR10#_MCMC.mat', {'pred': total_pred, 'pred_vectors':pred_vectors, 'label':labels_vector, 'low_level_vectors':low_level_vectors,
                            #    'high_level_vectors':high_level_vectors,  'Centers':Centers, 'prototype_feature':prototype_feature})