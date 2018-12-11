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
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

class PairwiseInteract(nn.Module):
    def __init__(self, get_force_layer_sizes=[2*1, 50, 50, 50, 20], apply_force_layer_sizes=[20, 50, 1], n_classes=2):
        assert get_force_layer_sizes[0] == 2*apply_force_layer_sizes[-1], 'need consistent state size'
        assert get_force_layer_sizes[-1] == apply_force_layer_sizes[0], 'need consistent force size'
        super(PairwiseInteract, self).__init__()
        self.state_dim = apply_force_layer_sizes[-1]
        self.force_dim = get_force_layer_sizes[-1]
        self.n_classes = n_classes
        self.get_force_modules = {}
        self.apply_force_modules = {}
        self.actors = [str(c) for c in range(self.n_classes)] + ['p' + str(c) for c in range(self.n_classes)]
        self.actees = [str(c) for c in range(self.n_classes)]
        for actor in self.actors:
            for actee in self.actees:
                self.get_force_modules[(actor, actee)] = MLP(get_force_layer_sizes)
                self.add_module('get_force_({},{})'.format(actor, actee), self.get_force_modules[(actor, actee)])
        for actee in self.actees:
            self.apply_force_modules[actee] = MLP(apply_force_layer_sizes)
            self.add_module('apply_force_{}'.format(actee), self.apply_force_modules[actee])

    def forward(self, obj_locs, prev_obj_locs):
        assert len(obj_locs) == self.n_classes, 'must have list of object locations for each object class'
        assert len(prev_obj_locs) == self.n_classes, 'must have list of object locations for each object class'
        forces = []
        preds = []
        for (c, actee) in enumerate(self.actees):
            forces.append(torch.zeros((len(obj_locs[c]), self.force_dim)))
        for (actor, actor_objs) in zip(self.actors, obj_locs+prev_obj_locs):
            for (c, (actee, actee_objs)) in enumerate(zip(self.actees, obj_locs)):
                combs = torch.stack([
                    actor_objs.reshape(1, -1).repeat(len(actee_objs), 1).flatten(),
                    actee_objs.reshape(-1, 1).repeat(1, len(actor_objs)).flatten(),
                ]).transpose(1, 0)
                comb_forces = self.get_force_modules[(actor, actee)](combs)
                forces[c] += torch.sum(
                    comb_forces.reshape(len(actee_objs), len(actor_objs), self.force_dim),
                    dim=1,
                )
        for (c, (actee, actee_objs)) in enumerate(zip(self.actees, obj_locs)):
            preds.append(
                self.apply_force_modules[actee](forces[c])
            )
        return preds
