from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gumbel_softmax import gumbel_softmax_sample

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

    kl_weight = 0.0
    reconstr_weight = 1.0
    learning_rate = 0.001

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(0)

    device = torch.device("cuda" if use_cuda else "cpu")

    enc = Encoder().to(device)
    dec = Decoder().to(device)
    params = {}
    for (k, v) in enc.named_parameters():
        params['enc.'+k.replace('__', '.')] = v
    for (k, v) in dec.named_parameters():
        params['dec.'+k.replace('__', '.')] = v
    optimizer = optim.Adam(params.values(), lr=learning_rate)

    data = h5py.File('data/obj_balls.h5', 'r')
    gt_pos = np.round(data['training']['positions'][49,[0,11,222,3333,4444]]*4).astype(np.uint8)
    #ims = np.zeros([5, 1, 64, 64], dtype=np.float32)
    #for i in range(5):
    #    for pos in gt_pos[i]:
    #        ims[i, 0] = cv2.circle(ims[i, 0], tuple(pos), 10, 1, -1)
    #ims = np.tile(ims, [1, 3, 1, 1]) - 0.5
    #ims_tensor = torch.tensor(ims, device=device)
    ims = np.tile(data['training']['features'][49,[0,11,222,3333,44444],:,:,0] - 0.5, [3,1,1,1]).transpose([1,0,2,3]).astype(np.float32)
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

    prior = np.array([(64*64-4)/(64*64.), 4/(64*64.)]).astype(np.float32)
    prior = np.reshape(prior, [1, -1, 1, 1])
    prior = torch.tensor(np.tile(prior, [1, 1, 64, 64]), device=device)
    eps = 1e-20

    enc.train()
    dec.train()
    for itr in range(10000):
        latent = enc(ims_tensor)
        samples = gumbel_softmax_sample(
            logits=latent.permute(0, 2, 3, 1),
            temperature=0.1,
        ).permute(0, 3, 1, 2)
        reconstr = dec(samples)

        optimizer.zero_grad()
        kl_loss = torch.mean(
            latent * (torch.log(latent+eps) - torch.log(prior+eps)),
        )
        reconstr_loss = torch.mean(
            (ims_tensor - reconstr)**2
        )
        kl_weight = (itr-1000)/1000. if itr > 1000 else 0
        loss = kl_weight*kl_loss + reconstr_weight*reconstr_loss
        print(itr, kl_weight*kl_loss.detach().cpu().numpy(), reconstr_weight*reconstr_loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
    for i in range(5):
        latent_im = latent.detach().cpu().numpy()[i, 0]
        samples_im = samples.detach().cpu().numpy()[i, 0]
        gt_im = gt_tensor.detach().cpu().numpy()[i, 0]
        input_im = ims_tensor.detach().cpu().numpy()[i, 0]+0.5
        reconstr_im = reconstr.detach().cpu().numpy()[i, 0]+0.5
        cv2.imwrite('data/latent_{}.png'.format(i), (255*np.clip(latent_im, 0, 1)).astype(np.uint8))
        cv2.imwrite('data/sampled_{}.png'.format(i), (255*np.clip(samples_im, 0, 1)).astype(np.uint8))
        # cv2.imwrite('gt.png', gt_im)
        # cv2.imwrite('im.png', input_im)
        cv2.imwrite('data/reconstr_{}.png'.format(i), (255*np.clip(reconstr_im, 0, 1)).astype(np.uint8))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
