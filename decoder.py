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

    kl_weight = 1.0
    reconstr_weight = 1.0
    learning_rate = 0.001
    mb_size = 20
    train_frac = 0.8
    val_frac = 0.1

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
    data_len = data['training']['features'].shape[1]

    prior = np.array([(64*64-4)/(64*64.), 4/(64*64.)]).astype(np.float32)
    prior = np.reshape(prior, [1, -1, 1, 1])
    prior = torch.tensor(np.tile(prior, [1, 1, 64, 64]), device=device)

    eps = 1e-20
    enc.train()
    dec.train()
    for epoch in range(100):
        data_inds = np.arange(int(data_len * train_frac))
        np.random.shuffle(data_inds)
        for mb in range(data_len // mb_size):
            mb_inds = sorted(data_inds[mb_size*mb : mb_size*(mb+1)])
            ims = np.tile(data['training']['features'][0,mb_inds,:,:,0] - 0.5, [3,1,1,1]).transpose([1,0,2,3]).astype(np.float32)
            ims_tensor = torch.tensor(ims, device=device)

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
            kl_weight = (epoch-10)/9. if epoch > 10 else 0
            loss = kl_weight*kl_loss + reconstr_weight*reconstr_loss
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(epoch, kl_weight*kl_loss.detach().cpu().numpy(), reconstr_weight*reconstr_loss.detach().cpu().numpy())
    print(epoch, kl_weight*kl_loss.detach().cpu().numpy(), reconstr_weight*reconstr_loss.detach().cpu().numpy())

    val_inds = np.arange(int(data_len * train_frac), int(data_len * (train_frac+val_frac)))
    np.random.shuffle(val_inds)
    for i in range(5):
        latent_im = latent.detach().cpu().numpy()[val_inds[i], 0]
        samples_im = samples.detach().cpu().numpy()[val_inds[i], 0]
        input_im = ims_tensor.detach().cpu().numpy()[val_inds[i], 0]+0.5
        reconstr_im = reconstr.detach().cpu().numpy()[val_inds[i], 0]+0.5
        cv2.imwrite('data/latent_{}.png'.format(i), (255*np.clip(latent_im, 0, 1)).astype(np.uint8))
        cv2.imwrite('data/sampled_{}.png'.format(i), (255*np.clip(samples_im, 0, 1)).astype(np.uint8))
        cv2.imwrite('data/im_{}.png'.format(i), (255*np.clip(input_im, 0, 1)).astype(np.uint8))
        cv2.imwrite('data/reconstr_{}.png'.format(i), (255*np.clip(reconstr_im, 0, 1)).astype(np.uint8))
