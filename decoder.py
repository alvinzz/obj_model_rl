from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Decoder(nn.Module):
    def __init__(self, channels=[16, 16, 3], obj_classes=2):
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
                    F.elu(
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

if __name__ == '__main__':
    import numpy as np
    import h5py
    import cv2
    from encoder import Encoder

    kl_weight = 1.0
    reconstr_weight = 1.0
    learning_rate = 0.001

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(0)

    device = torch.device("cuda" if use_cuda else "cpu")

    enc = Encoder().to(device)
    dec = Decoder([9, 36, 3]).to(device)
    params = {}
    for (k, v) in enc.named_parameters():
        params['enc.'+k.replace('__', '.')] = v
    for (k, v) in dec.named_parameters():
        params['dec.'+k.replace('__', '.')] = v
    optimizer = optim.Adam(params.values(), lr=learning_rate)

    data = h5py.File('/home/alvin/Windows/Downloads/obj_balls.h5', 'r')
    gt_pos = np.round(data['training']['positions'][:5,0]*4).astype(np.uint8)
    # ims = np.zeros([1, 1, 64, 64], dtype=np.float32)
    # for i in range(1):
    #     for pos in gt_pos[i]:
    #         ims[i, 0] = cv2.circle(ims[i, 0], tuple(pos), 5, 1, -1)
    # ims = np.tile(ims, [1, 3, 1, 1]) - 0.5
    # ims_tensor = torch.tensor(ims, device=device)
    ims = np.tile(data['training']['features'][:5,0,:,:,0] - 0.5, [3,1,1,1]).transpose([1,0,2,3]).astype(np.float32)
    ims_tensor = torch.tensor(ims, device=device)

    gt_latent = np.zeros([5, 2, 64, 64], dtype=np.float32)
    gt_latent[:, 0, :, :] = 1.
    for i in range(5):
        for pos in gt_pos[i]:
            gt_latent[i, 0, pos[1], pos[0]] = 0.
            gt_latent[i, 1, pos[1], pos[0]] = 1.
    gt_tensor = torch.tensor(gt_latent, device=device)

    # im = np.zeros((64, 64), dtype=np.float32)
    # for pos in gt_pos[0]:
    #     cv2.circle(im, tuple(pos), 5, 1)
    # cv2.imshow('gt_im', im)
    # cv2.imshow('im', ims[0].transpose([1,2,0]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    prior = np.array([60./64, 4./64]).astype(np.float32)
    prior = np.reshape(prior, [1, -1, 1, 1])
    prior = torch.tensor(np.tile(prior, [1, 1, 64, 64]))
    eps = 1e-20

    enc.train()
    dec.train()
    for itr in range(50):
        # latent = enc(ims_tensor)
        latent = gt_tensor
        reconstr = dec(latent)

        optimizer.zero_grad()
        # kl_loss = torch.mean(
        #     latent * (torch.log(latent+eps) - torch.log(prior+eps)),
        # )
        kl_loss = torch.mean(
            latent[:,0] * (torch.log(latent[:,0]+eps) - torch.log(gt_tensor[:,0]+eps)),
        )
        reconstr_loss = torch.mean(
            (ims_tensor - reconstr)**2
        )
        loss = kl_weight*kl_loss + reconstr_weight*reconstr_loss
        print(itr, loss.detach().numpy())
        loss.backward()
        optimizer.step()
    for i in range(5):
        latent_im = latent.detach().numpy()[i, 0]
        cv2.imshow('latent', latent_im)
        # cv2.imshow('gt', gt_tensor.detach().numpy()[0, 0])
        cv2.imshow('im', ims_tensor.detach().numpy()[i, 0]+0.5)
        cv2.imshow('reconstr', reconstr.detach().numpy()[i, 0]+0.5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
