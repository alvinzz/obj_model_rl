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
from action_dyn_model import PairwiseInteract
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
    dataset = pickle.load(open('data/multiagent_dyn_latent.pkl', 'rb'))

    for epoch in range(10000):
        dataset_len = 0
        tot_loss = 0
        for (prev, cur, targ, action, cost) in tqdm(dataset[:]):
            if all([len(k) for k in prev]):
                prev = [k.cpu().type(torch.FloatTensor) for k in prev]
                cur = [k.cpu().type(torch.FloatTensor) for k in cur]
                targ = [k.cpu().type(torch.FloatTensor) for k in targ]
                action = action.cpu().type(torch.FloatTensor)
                cost = cost.cpu().type(torch.FloatTensor)

                pred, pred_cost = dyn_model.forward(cur, prev, action)
                pred_loss = torch.mean(torch.abs(rnn_utils.pad_sequence(pred)-rnn_utils.pad_sequence(targ)))
                cost_loss = torch.squeeze((cost - pred_cost)**2)
                loss = pred_loss + cost_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                dataset_len += 1
                tot_loss += loss.detach().cpu().numpy()
        if epoch % 1 == 0:
            params = {k: v.data.cpu().numpy() for (k, v) in dyn_model.named_parameters()}
            pickle.dump(params, open('data/multiagent_dyn_model.pkl', 'wb'))
            print(tot_loss / dataset_len)
    print(tot_loss / dataset_len)
