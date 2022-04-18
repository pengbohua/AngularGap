import os
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='arguments to compute prediction depth for each data sample')
parser.add_argument('--result_dir', default='./cl_results_wsgn', type=str, help='directory to save ckpt and results')
parser.add_argument('--arch', default='resnet', type=str, help='arch for prediction depth')
parser.add_argument('--knn_k', default=30, type=int, help='k nearest neighbors of knn classifier')
parser.add_argument('--num_samples', default=10000, type=int, help='number samples of current dst')

args = parser.parse_args()

seeds = [1111, 2222, 3333, 4444, 5555, 6666]

arch = args.arch
pd_dir = args.result_dir

print('computing prediction depth in train split')
pd_train_split = np.zeros((len(seeds), args.num_samples))
for i, sd in enumerate(seeds):

            f = os.path.join(pd_dir, '{}train_seed{}_f_trainpd.pkl'.format(arch, sd))
            with open(f, 'r') as p:
                pd_dict = json.load(p)
            for k, v in pd_dict.items():
                pd_train_split[i, int(k)] = v[0]
            f = os.path.join(pd_dir, '{}train_seed{}_fflip_trainpd.pkl'.format(arch, sd))
            with open(f, 'r') as p:
                pd_dict = json.load(p)
            for k, v in pd_dict.items():
                pd_train_split[i, int(k)] = v[0]

print(pd_train_split.shape)
pd_train_split_avg = pd_train_split.mean(0)
train_split_small_pds = np.where((pd_train_split_avg >1) & (pd_train_split_avg <= 2))[0]

print('computing prediction depth in test split')
pd_test_split = np.zeros((len(seeds), args.num_samples))
for i, sd in enumerate(seeds):
            f = os.path.join(pd_dir, '{}_seed{}_f_test_pd.pkl'.format(arch, sd))
            with open(f, 'r') as p:
                pd_dict = json.load(p)
            for k, v in pd_dict.items():
                pd_test_split[i, int(k)] = v[0]

            f = os.path.join(pd_dir, '{}_seed{}_fflip_test_pd.pkl'.format(arch, sd))
            with open(f, 'r') as p:
                pd_dict = json.load(p)
            for k, v in pd_dict.items():
                pd_test_split[i, int(k)] = v[0]

def show_sample(index, dataset):
    img, _ = dataset[index]
    img = img.permute(1,2,0).numpy()
    plt.imshow(img)
    plt.savefig('./easy_samples/img{}.png'.format(index))
    plt.show()

pd_test_split_avg = pd_test_split.mean(0)


H, x_edges, y_edges = np.histogram2d(pd_test_split_avg - 1, pd_train_split_avg - 1, bins=(np.linspace(0, 9, 50), np.linspace(0, 9, 50)))
plt.figure()
H[H < 1e-7] = np.nan
H = H.T
X, Y = np.meshgrid(x_edges, y_edges)
plt.pcolormesh(X, Y, H)
plt.xlabel('validation split prediction depth')
plt.ylabel('train split prediction depth')
plt.colorbar()
plt.savefig(os.path.join(pd_dir, '/prediction_depth_12resnet{}.png').format(args.knn_k))
plt.show()