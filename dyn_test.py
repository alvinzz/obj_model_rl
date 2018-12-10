from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
use_cuda = torch.cuda.is_available()

class MLP(nn.Module):
    def __init__(self, layer_sizes=[1, 100, 100, 100, 2]):
        # num_obj_classes includes background class
        super(MLP, self).__init__()
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.add_module('layer_{}'.format(i), self.layers[i])

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.leaky_relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

mb_size = 1000
grid_size = 20 #100
max_shift = 2
force_size = 25
device = torch.device("cuda" if use_cuda else "cpu")

prev_force_model = MLP([2, 100, 100, force_size]).to(device)
render_model = MLP([force_size, 250, 250, (2*max_shift+1)**2]).to(device)
params = list(prev_force_model.parameters()) + list(render_model.parameters())
for p in params:
    try:
        nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    except:
        pass
optimizer = optim.Adam(params, lr=0.001)
prev_force_model.train()
render_model.train()

diff_combs = np.dstack(np.meshgrid(np.arange(grid_size), np.arange(grid_size))).reshape(-1, 2)
diff_combs = np.stack((diff_combs[:, 1], diff_combs[:, 0]), axis=1)
import itertools
diff_combs = np.array(list(itertools.product(diff_combs, diff_combs)))
diff_combs = diff_combs[:,1,:] - diff_combs[:,0,:]
diff_combs = torch.Tensor(diff_combs).to(device)

def reflect_ind(ind, lower, upper):
    if ind < lower:
        return lower + (lower - ind)
    if ind > upper:
        return upper - (ind - upper)
    return ind
reflect_ind = np.vectorize(reflect_ind)

for itr in range(10000000):
    inds = np.random.randint(2*max_shift, grid_size-1-2*max_shift, size=(mb_size, 2), dtype=np.uint8)
    shifts = np.random.randint(-max_shift, max_shift+1, size=(mb_size, 2), dtype=np.int8)
    
    prev = np.zeros((mb_size, grid_size, grid_size))
    cur = np.zeros((mb_size, grid_size, grid_size))
    targ = np.zeros((mb_size, grid_size, grid_size))
    
    prev[np.arange(mb_size), inds[:,0], inds[:,1]] = 1
    cur[np.arange(mb_size), inds[:,0]+shifts[:,0], inds[:,1]+shifts[:,1]] = 1
    targ[np.arange(mb_size), inds[:,0]+2*shifts[:,0], inds[:,1]+2*shifts[:,1]] = 1
    
    prev = Variable(torch.Tensor(prev), requires_grad=True).to(device)
    cur = Variable(torch.Tensor(cur), requires_grad=True).to(device)
    targ = Variable(torch.Tensor(targ), requires_grad=False).to(device)
    
    prev_forces = prev_force_model(diff_combs.type(torch.FloatTensor).to(device))
    render = []
    for mb_ind in range(mb_size):
        cur_locs = cur[mb_ind].nonzero()
        prev_locs = prev[mb_ind].nonzero()
        render_mb = torch.zeros((grid_size+2*max_shift, grid_size+2*max_shift), dtype=torch.float32).to(device)
        for cur_loc in cur_locs:
            force_on_cur_loc = torch.zeros(force_size).type(torch.FloatTensor).to(device)
            for prev_loc in prev_locs:
                force = prev_forces[prev_loc[0]*1000 + prev_loc[1]*100 + cur_loc[0]*10 + cur_loc[1]]
                force_on_cur_loc += force
            shift_pred = render_model(force_on_cur_loc).reshape(2*max_shift+1, 2*max_shift+1)
            render_mb[cur_loc[0]+max_shift-max_shift:cur_loc[0]+max_shift+max_shift+1, cur_loc[1]+max_shift-max_shift:cur_loc[1]+max_shift+max_shift+1] += shift_pred
        render.append(render_mb)
    render = torch.stack(render, dim=0)
    render = render[:, max_shift:max_shift+grid_size, max_shift:max_shift+grid_size]
    
    optimizer.zero_grad()
    loss = 100*torch.mean(
        (render - targ)**2,
    )
    loss.backward()
    optimizer.step()
    
    if itr % 100 == 0:
#        print(loss, prev[0], cur[0], torch.round(render[0]))
        print(loss)
    if itr % 1000 == 0:
        import cv2
        cv2.imwrite('preds{}.jpg'.format(itr), (255*np.stack([prev[0].detach().cpu().numpy(), cur[0].detach().cpu().numpy(), render[0].detach().cpu().numpy()], axis=2)).astype(np.uint8))
        cv2.imwrite('targ{}.jpg'.format(itr), (255*np.stack([prev[0].detach().cpu().numpy(), cur[0].detach().cpu().numpy(), targ[0].detach().cpu().numpy()], axis=2)).astype(np.uint8))
print(loss)
