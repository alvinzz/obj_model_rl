import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import h5py
import cv2

from encoder import Encoder
from decoder import Decoder
from gumbel_softmax import gumbel_softmax_sample

import time
from tqdm import tqdm
import pickle

def test_autoencoder():
    kl_weight = 1.0
    reconstr_weight = 1.0
    learning_rate = 0.01
    mb_size = 64

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if use_cuda else "cpu")

    #enc = Encoder([3,96,96,3], [0,0,0,3], 3).to(device)
    enc = Encoder([3,96,96,3], [0,1,0,3], 4).to(device)
    dec = Decoder([9,36,36,3], 4).to(device)
    params = {}
    for (k, v) in enc.named_parameters():
        params['enc.'+k.replace('__', '.')] = v
    for (k, v) in dec.named_parameters():
        params['dec.'+k.replace('__', '.')] = v
    optimizer = optim.Adam(params.values(), lr=learning_rate)

    data = h5py.File('data/obj_balls.h5', 'r')
    print('extracting datasets to numpy...')
    # train_data = data['training']['features'][:1,:]
    # val_data = data['validation']['features'][:1,:]
    train_data = data['training']['groups'][:1,:]
    val_data = data['validation']['groups'][:1,:]
    print('done!')

    #prior = np.array([(64*64-4)/(64*64.), 2/(64*64.), 2/(64*64.)]).astype(np.float32)
    #prior = np.array([(64*64-2-63*4)/(64*64.), 1/(64*64.), 1/(64*64.), 63*4/(64*64.)]).astype(np.float32)
    prior = np.array([(64*64-3)/(64*64.), 1/(64*64.), 1/(64*64.), 1/(64*64.)]).astype(np.float32)
    prior = np.reshape(prior, [1, -1, 1, 1])
    prior = torch.tensor(np.tile(prior, [1, 1, 64, 64]), device=device)

    eps = 1e-20
    enc.train()
    dec.train()
    for epoch in range(50): #10 #30
        mb_inds = np.arange(train_data.shape[1])
        np.random.shuffle(mb_inds)
        for mb in tqdm(range(len(mb_inds) // mb_size)):
        # for mb in range(len(mb_inds) // mb_size):
            # ims = np.tile(train_data[0,mb_inds[mb*mb_size:(mb+1)*mb_size],:,:,0] - 0.5, [3,1,1,1]).transpose([1,0,2,3]).astype(np.float32)
            ims = np.zeros((mb_size, 3, 64, 64), dtype=np.float32)
            locs = np.where(train_data[0, mb_inds[mb*mb_size:(mb+1)*mb_size]] == 2)
            ims[locs[0], np.zeros_like(locs[3]), locs[1], locs[2]] = 1.
            locs = np.where(train_data[0, mb_inds[mb*mb_size:(mb+1)*mb_size]] == 1)
            ims[locs[0], 2*np.ones_like(locs[3]), locs[1], locs[2]] = 1.
            for x in [0, 63]:
                for y in range(64):
                    ims[:, :, x, y] = 1.
            for y in [0, 63]:
                for x in range(64):
                    ims[:, :, x, y] = 1.
            ims -= 0.5
            ims_tensor = torch.tensor(ims, device=device)

            latent = enc(ims_tensor)
            samples = gumbel_softmax_sample(
                logits=latent.permute(0, 2, 3, 1),
                temperature=0.1,
            ).permute(0, 3, 1, 2)
            reconstr = dec(samples)
            # reconstr = dec(latent)

            optimizer.zero_grad()
            kl_loss = torch.mean(
                latent * (torch.log(latent+eps) - torch.log(prior+eps)),
            )
            reconstr_loss = torch.mean(
                (ims_tensor - reconstr)**2
            )
            # kl_weight = epoch / 3. # original
            kl_weight = epoch / 3000.
            # kl_weight = (epoch-1000) / 9000. if epoch > 1000 else 0
            loss = kl_weight*kl_loss + reconstr_weight*reconstr_loss
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(epoch, kl_weight*kl_loss.detach().cpu().numpy(), reconstr_weight*reconstr_loss.detach().cpu().numpy())
            val_inds = np.arange(val_data.shape[1])
            np.random.shuffle(val_inds)
            for val_ind in val_inds[:5]:
                # ims = np.tile(val_data[0,val_ind:val_ind+1,:,:,0] - 0.5, [3,1,1,1]).transpose([1,0,2,3]).astype(np.float32)
                ims = np.zeros((mb_size, 3, 64, 64), dtype=np.float32)
                locs = np.where(val_data[0, val_ind:val_ind+1] == 2)
                ims[locs[0], np.zeros_like(locs[3]), locs[1], locs[2]] = 1.
                locs = np.where(val_data[0, val_ind:val_ind+1] == 1)
                ims[locs[0], 2*np.ones_like(locs[3]), locs[1], locs[2]] = 1.
                for x in [0, 63]:
                    for y in range(64):
                        ims[:, :, x, y] = 1.
                for y in [0, 63]:
                    for x in range(64):
                        ims[:, :, x, y] = 1.
                ims -= 0.5
                ims_tensor = torch.tensor(ims, device=device)

                latent = enc(ims_tensor)
                samples = gumbel_softmax_sample(
                    logits=latent.permute(0, 2, 3, 1),
                    temperature=0.1,
                ).permute(0, 3, 1, 2)
                reconstr = dec(samples)

                logdir = 'autoencoder_exp_kl3000_' + time.strftime("%d-%m-%Y_%H-%M")
                try:
                    os.system('mkdir data/{}'.format(logdir))
                except Exception as e:
                    pass

                latent_im = (latent.detach().cpu().numpy()[0, :]).transpose([1,2,0])
                samples_im = (samples.detach().cpu().numpy()[0, :]).transpose([1,2,0])
                input_im = (ims_tensor.detach().cpu().numpy()[0, :]+0.5).transpose([1,2,0])
                reconstr_im = (reconstr.detach().cpu().numpy()[0, :]+0.5).transpose([1,2,0])
                pickle.dump(latent_im, open('data/{}/latent_{}.pkl'.format(logdir, val_ind), 'wb'))
                pickle.dump(samples_im, open('data/{}/sampled_{}.pkl'.format(logdir, val_ind), 'wb'))
                pickle.dump(reconstr_im, open('data/{}/reconstr_{}.pkl'.format(logdir, val_ind), 'wb'))
                pickle.dump(input_im, open('data/{}/im_{}.pkl'.format(logdir, val_ind), 'wb'))
                cv2.imwrite('data/{}/latent_{}.png'.format(logdir, val_ind), (255*np.clip(latent_im, 0, 1)).astype(np.uint8)[:,:,-3:])
                cv2.imwrite('data/{}/sampled_{}.png'.format(logdir, val_ind), (255*np.clip(samples_im, 0, 1)).astype(np.uint8)[:,:,-3:])
                cv2.imwrite('data/{}/im_{}.png'.format(logdir, val_ind), (255*np.clip(input_im, 0, 1)).astype(np.uint8))
                cv2.imwrite('data/{}/reconstr_{}.png'.format(logdir, val_ind), (255*np.clip(reconstr_im, 0, 1)).astype(np.uint8))

    print(epoch, kl_weight*kl_loss.detach().cpu().numpy(), reconstr_weight*reconstr_loss.detach().cpu().numpy())

    for val_ind in val_inds[:5]:
        # ims = np.tile(val_data[0,val_ind:val_ind+1,:,:,0] - 0.5, [3,1,1,1]).transpose([1,0,2,3]).astype(np.float32)
        ims = np.zeros((mb_size, 3, 64, 64), dtype=np.float32)
        locs = np.where(val_data[0, val_ind:val_ind+1] == 2)
        ims[locs[0], np.zeros_like(locs[3]), locs[1], locs[2]] = 1.
        locs = np.where(val_data[0, val_ind:val_ind+1] == 1)
        ims[locs[0], 2*np.ones_like(locs[3]), locs[1], locs[2]] = 1.
        for x in [0, 63]:
            for y in range(64):
                ims[:, :, x, y] = 1.
        for y in [0, 63]:
            for x in range(64):
                ims[:, :, x, y] = 1.
        ims -= 0.5
        ims_tensor = torch.tensor(ims, device=device)

        latent = enc(ims_tensor)
        samples = gumbel_softmax_sample(
            logits=latent.permute(0, 2, 3, 1),
            temperature=0.1,
        ).permute(0, 3, 1, 2)
        reconstr = dec(samples)

        logdir = 'autoencoder_exp_kl3000_' + time.strftime("%d-%m-%Y_%H-%M")
        try:
            os.system('mkdir data/{}'.format(logdir))
        except Exception as e:
            pass

        latent_im = (latent.detach().cpu().numpy()[0, :]).transpose([1,2,0])
        samples_im = (samples.detach().cpu().numpy()[0, :]).transpose([1,2,0])
        input_im = (ims_tensor.detach().cpu().numpy()[0, :]+0.5).transpose([1,2,0])
        reconstr_im = (reconstr.detach().cpu().numpy()[0, :]+0.5).transpose([1,2,0])
        pickle.dump(latent_im, open('data/{}/latent_{}.pkl'.format(logdir, val_ind), 'wb'))
        pickle.dump(samples_im, open('data/{}/sampled_{}.pkl'.format(logdir, val_ind), 'wb'))
        pickle.dump(reconstr_im, open('data/{}/reconstr_{}.pkl'.format(logdir, val_ind), 'wb'))
        pickle.dump(input_im, open('data/{}/im_{}.pkl'.format(logdir, val_ind), 'wb'))
        cv2.imwrite('data/{}/latent_{}.png'.format(logdir, val_ind), (255*np.clip(latent_im, 0, 1)).astype(np.uint8)[:,:,-3:])
        cv2.imwrite('data/{}/sampled_{}.png'.format(logdir, val_ind), (255*np.clip(samples_im, 0, 1)).astype(np.uint8)[:,:,-3:])
        cv2.imwrite('data/{}/im_{}.png'.format(logdir, val_ind), (255*np.clip(input_im, 0, 1)).astype(np.uint8))
        cv2.imwrite('data/{}/reconstr_{}.png'.format(logdir, val_ind), (255*np.clip(reconstr_im, 0, 1)).astype(np.uint8))

if __name__ == '__main__':
    test_autoencoder()
