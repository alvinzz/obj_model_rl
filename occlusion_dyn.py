from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools
import torch.nn.utils.rnn as rnn_utils
from dyn_model import PairwiseInteract
import pickle
from tqdm import tqdm

use_cuda = torch.cuda.is_available()

if __name__ == '__main__':
    n_classes = 2
    dyn_model = PairwiseInteract(n_classes=n_classes)
    for (name, param) in dyn_model.named_parameters():
        if name.endswith('weight'):
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
    optimizer = optim.Adam(dyn_model.parameters(), lr=0.000001)
    dataset = pickle.load(open('data/occlusion_latent.pkl', 'rb'))

    for epoch in range(10000):
        dataset_len = 0
        tot_loss = 0
        for (prev, cur, targ) in tqdm(dataset[:]):
            if all([len(k) for k in prev]):
                prev = [k.cpu().type(torch.FloatTensor) for k in prev]
                cur = [k.cpu().type(torch.FloatTensor) for k in cur]
                targ = [k.cpu().type(torch.FloatTensor) for k in targ]
                pred = dyn_model.forward(cur, prev)
                loss = torch.mean((rnn_utils.pad_sequence(pred)-rnn_utils.pad_sequence(targ))**2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                dataset_len += 1
                tot_loss += loss.detach().cpu().numpy()
        if epoch % 1 == 0:
            params = {k: v.data.cpu().numpy() for (k, v) in dyn_model.named_parameters()}
            pickle.dump(params, open('data/occusion_dyn_model.pkl', 'wb'))
            print(tot_loss / dataset_len)
    print(tot_loss / dataset_len)
