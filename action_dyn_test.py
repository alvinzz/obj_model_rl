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

use_cuda = torch.cuda.is_available()

if __name__ == '__main__':
    #interact = PairwiseInteract(n_classes=2)
    #obj_locs = [torch.Tensor([[5],[4]]),torch.Tensor([[1],[2],[3]])]
    #prev_obj_locs = [torch.Tensor([[5],[4]]),torch.Tensor([[1],[2],[3]])]
    #preds = interact.forward(obj_locs, prev_obj_locs)
    #print(preds)

    #interact = PairwiseInteract(n_classes=2)
    #obj_locs = [torch.Tensor([[5,5],[4,4]]),torch.Tensor([[1,1],[2,2],[3,3]])]
    #prev_obj_locs = [torch.Tensor([[5,5],[4,4]]),torch.Tensor([[1,1],[2,2],[3,3]])]
    #action = torch.Tensor([1, 1])
    #preds, cost = interact.forward(obj_locs, prev_obj_locs, action)
    #print(preds, cost)
    #print(1/0)

    def disp(cur, prev, pred):
        cur = cur[0].detach().cpu().numpy().flatten().astype(np.int32)
        prev = prev[0].detach().cpu().numpy().flatten().astype(np.int32)
        pred = np.round(pred[0].detach().cpu().numpy().flatten()).astype(np.int32)
        import cv2
        im = np.zeros((3, 7), dtype=np.float32)
        for i in range(len(cur)):
            im[0, prev[i]] = (1+i)/len(cur)
            im[1, cur[i]] = (1+i)/len(cur)
            try:
                im[2, pred[i]] = (1+i)/len(cur)
            except:
                pass
        im = cv2.resize(im, (0,0), fx=10, fy=10)
        cv2.imshow('im', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    dyn_model = PairwiseInteract(
        action_force_layer_sizes=[1, 100, 100, 1],
        get_force_layer_sizes=[2*1, 100, 100, 1],
        apply_force_layer_sizes=[1+1, 100, 100, 1],
        cost_layer_sizes=[1, 100, 100, 1],
        n_classes=1
    )
    for (name, param) in dyn_model.named_parameters():
        if name.endswith('weight'):
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
    optimizer = optim.Adam(dyn_model.parameters(), lr=0.0001)
    for epoch in range(10000):
        n_objs = np.random.randint(2, 3)

        prev = np.random.randint(0, 7, size=n_objs)
        #prev = np.array([np.random.randint(2, 3), np.random.randint(7, 8)])
        shifts = np.random.randint(-1, 2, size=n_objs)
        action = np.random.randint(-1, 2)
        cur = prev + shifts
        cost = np.abs(cur[0] - cur[1])
        for i in range(len(cur)):
            if cur[i] < 0:
                cur[i] = -cur[i]
                shifts[i] = -shifts[i]
            if cur[i] >= 7:
                cur[i] = 6-(cur[i]-6)
                shifts[i] = -shifts[i]
        targ = cur + shifts + action
        for i in range(len(targ)):
            if targ[i] < 0:
                targ[i] = -targ[i]
            if targ[i] >= 7:
                targ[i] = 6-(targ[i]-6)

        prev = prev.reshape(1, n_objs, 1)
        prev = [torch.Tensor(n) for n in prev]
        cur = cur.reshape(1, n_objs, 1)
        cur = [torch.Tensor(n) for n in cur]
        targ = targ.reshape(1, n_objs, 1)
        targ = [torch.Tensor(n) for n in targ]
        cost = torch.Tensor([cost])
        action = torch.Tensor([action])

        pred, pred_cost = dyn_model.forward(cur, prev, action)

        pred_loss = torch.mean(torch.abs(rnn_utils.pad_sequence(pred)-rnn_utils.pad_sequence(targ)))
        cost_loss = (cost-pred_cost)**2
        loss = pred_loss + cost_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(pred_loss, cost_loss)
            print(action)
            disp(cur, prev, pred)
    print(pred_loss, cost_loss)
    print(action)
    disp(cur, prev, pred)
