from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, channels=[3, 16, 16, 2], scale_feats=[0, 0, 0, 2], obj_classes=2):
        # num_obj_classes includes background class
        assert len(channels) == len(scale_feats), \
            'length of list of per-scale features must match length of list of channels'
        assert channels[-1] == scale_feats[-1], \
            'wasting computation if more channels than scale feats at last layer'
        assert scale_feats[0] == 0, \
            'shouldn\'t have feature channels for input image'
        assert sum(scale_feats) >= obj_classes, \
            'should have at least one feature channel per object class'
        super(Encoder, self).__init__()

        self.channels = channels
        self.scale_feats = scale_feats
        self.obj_classes = obj_classes

        self.conv_ops = []
        self.point_ops = []
        for i in range(len(self.channels)-1):
            self.point_ops.append(
                nn.Conv2d(
                    in_channels=self.channels[i], out_channels=self.channels[i+1],
                    kernel_size=1, stride=1, padding=0, dilation=1,
                    groups=1, bias=True,
                )
            )
            self.add_module('point_{}'.format(i), self.point_ops[i])
            self.conv_ops.append(
                nn.Conv2d(
                    in_channels=self.channels[i+1], out_channels=self.channels[i+1],
                    kernel_size=3, stride=1, padding=0, dilation=3**i,
                    groups=self.channels[i+1], bias=True,
                )
            )
            self.add_module('conv_{}'.format(i), self.conv_ops[i])

        self.feats_to_classes_op = nn.Conv2d(
            in_channels=sum(self.scale_feats), out_channels=obj_classes,
            kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bias=False,
        )

    def forward(self, x):
        self.layers = [x]
        self.feats = []
        for i in range(len(self.channels)-1):
            self.layers.append(
                F.relu(
                    self.conv_ops[i](
                        self.point_ops[i](
                            nn.ReflectionPad2d(3**i)(
                                self.layers[-1]
                            )
                        )
                    )
                )
            )
            if self.scale_feats[i+1] > 0:
                self.feats.append(self.layers[-1][:, -self.scale_feats[i+1]:, :, :])
        self.feats = torch.cat(self.feats, dim=1)
        self.obj_probs = nn.Softmax(dim=1)(
            self.feats_to_classes_op(self.feats)
        )
        return self.obj_probs
