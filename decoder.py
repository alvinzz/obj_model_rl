from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gumbel_softmax import gumbel_softmax_sample
from tqdm import tqdm

class Decoder(nn.Module):
    def __init__(self, obj_classes=2):
        # num_obj_classes includes background class
        super(Decoder, self).__init__()

        self.obj_classes = obj_classes

        self.conv_ops = []
        for obj_class in range(self.obj_classes):
            self.conv_ops.append(
                nn.ConvTranspose2d(
                    in_channels=1, out_channels=3,
                    kernel_size=27, stride=1, padding=13, dilation=1,
                    groups=1, bias=True,
                )
            )
            self.add_module('obj_{}__conv_0'.format(obj_class), self.conv_ops[obj_class])

        self.background = torch.nn.Parameter(torch.zeros([1, 3, 64, 64], dtype=torch.float32), requires_grad=True)

        self.combine_classes_op = nn.Conv2d(
            in_channels=3*(self.obj_classes+1), out_channels=3,
            kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bias=True,
        )

    def forward(self, x):
        self.obj_inputs = [x[:, obj_class:obj_class+1, :, :] for obj_class in range(self.obj_classes)]
        self.obj_outputs = [self.background.repeat(x.shape[0], 1, 1, 1)]
        for obj_class in range(self.obj_classes):
            self.obj_outputs.append(
                self.conv_ops[obj_class](
                    self.obj_inputs[obj_class]
                )
            )

        self.classes = torch.cat(self.obj_outputs, dim=1)
        self.objects = self.combine_classes_op(self.classes)
        return self.objects
