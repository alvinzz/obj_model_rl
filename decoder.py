from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gumbel_softmax import gumbel_softmax_sample
from tqdm import tqdm

class Decoder(nn.Module):
    def __init__(self, channels=[3, 36, 36, 3], obj_classes=2, n_reps=[1, 1, 1]):
        assert channels[0] == obj_classes + 1, 'channels[0] = background(1) + obj_classes'
        super(Decoder, self).__init__()

        self.channels = channels
        self.obj_classes = obj_classes
        self.n_reps = n_reps

        self.point_ops = []
        self.conv_ops = []
        for i in range(len(self.channels)-1):
            scaling = 3**(len(channels) - 2 - i)
            self.conv_ops.append([])
            self.point_ops.append([])
            for r in range(self.n_reps[i]):
                self.conv_ops[i].append(
                    nn.ConvTranspose2d(
                        in_channels=self.channels[i+1], out_channels=self.channels[i+1],
                        kernel_size=3, stride=1, padding=scaling, dilation=scaling,
                        groups=self.channels[i+1], bias=True,
                    )
                )
                self.add_module('conv_{}_{}'.format(i, r), self.conv_ops[i][r])

                if r == 0:
                    self.point_ops[i].append(
                        nn.Conv2d(
                            in_channels=self.channels[i], out_channels=self.channels[i+1],
                            kernel_size=1, stride=1, padding=0, dilation=1,
                            groups=1, bias=True,
                        )
                    )
                else:
                    self.point_ops[i].append(
                        nn.Conv2d(
                            in_channels=self.channels[i+1], out_channels=self.channels[i+1],
                            kernel_size=1, stride=1, padding=0, dilation=1,
                            groups=1, bias=True,
                        )
                    )
                self.add_module('point_{}_{}'.format(i, r), self.point_ops[i][r])

    def forward(self, x):
        x = torch.cat((torch.zeros_like(x[:, :1, :, :], dtype=torch.float32), x), dim=1)
        self.layers = [x]
        for i in range(len(self.channels)-1):
            for r in range(self.n_reps[i]):
                self.layers.append(
                    #F.leaky_relu(
                    identity(
                        self.conv_ops[i][r](
                            self.point_ops[i][r](
                                self.layers[-1]
                            )
                        )
                    )
                )
        self.objects = self.layers[-1]
        return self.objects

def identity(x):
    return x
