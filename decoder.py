from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gumbel_softmax import gumbel_softmax_sample
from tqdm import tqdm

class Decoder(nn.Module):
    def __init__(self, channels=[9, 36, 36, 3], obj_classes=2):
        # num_obj_classes includes background class
        super(Decoder, self).__init__()

        self.channels = channels
        self.obj_classes = obj_classes

        self.point_ops = {}
        self.conv_ops = {}
        for obj_class in range(self.obj_classes):
            self.point_ops[obj_class] = []
            self.conv_ops[obj_class] = []
            for i in range(len(self.channels)-1):
                scaling = 3**(len(channels) - 2 - i)
                self.conv_ops[obj_class].append(
                    nn.ConvTranspose2d(
                        in_channels=self.channels[i], out_channels=self.channels[i],
                        kernel_size=3, stride=1, padding=scaling, dilation=scaling,
                        groups=self.channels[i], bias=True,
                    )
                )
                self.add_module('obj_{}__conv_{}'.format(obj_class, i), self.conv_ops[obj_class][i])

                self.point_ops[obj_class].append(
                    nn.ConvTranspose2d(
                        in_channels=self.channels[i], out_channels=self.channels[i+1],
                        kernel_size=1, stride=1, padding=0, dilation=1,
                        groups=1, bias=True,
                    )
                )
                self.add_module('obj_{}__point_{}'.format(obj_class, i), self.point_ops[obj_class][i])


        self.combine_classes_op = nn.ConvTranspose2d(
            in_channels=self.channels[-1]*self.obj_classes, out_channels=self.channels[-1],
            kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bias=True,
        )

    def forward(self, x):
        self.class_layer_dict = {
            obj_class: [x[:, obj_class:obj_class+1, :, :].repeat(1, self.channels[0], 1, 1)]
            for obj_class in range(self.obj_classes)
        }
        for obj_class in range(self.obj_classes):
            for i in range(len(self.channels)-1):
                self.class_layer_dict[obj_class].append(
                    identity(
                        self.point_ops[obj_class][i](
                            self.conv_ops[obj_class][i](
                                self.class_layer_dict[obj_class][-1]
                            )
                        )
                    )
                )
        self.classes = torch.cat(
            [
                self.class_layer_dict[obj_class][-1]
                for obj_class in range(self.obj_classes)
            ],
            dim=1,
        )
        self.objects = self.combine_classes_op(self.classes)
        return self.objects

def identity(x):
    return x
