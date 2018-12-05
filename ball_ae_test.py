import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

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

    data = h5py.File('data/obj_balls.h5', 'r')
    print('extracting datasets to numpy...')
    train_data = data['training']['groups'][:1,:6400].transpose([1,0,2,3,4])
    val_data = data['validation']['groups'][:1,:5].transpose([1,0,2,3,4])
    print('done!')

    train_dataset = ObjDataset(train_data, device)
    val_dataset = ObjDataset(val_data, device)
    train_dataloader = DataLoader(train_dataset, batch_size=mb_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

    enc = Encoder([3,96,96,2], [0,0,0,2], 2).to(device)
    dec = Decoder([9,36,36,3], 2).to(device)
    params = {}
    for (k, v) in enc.named_parameters():
        params['enc.'+k.replace('__', '.')] = v
    for (k, v) in dec.named_parameters():
        params['dec.'+k.replace('__', '.')] = v
    #saved_weights = pickle.load(open('data/ae_kl_100_04-12-2018_01-15/2/params.pkl', 'rb'))
    #for (k,v) in saved_weights.items():
    #    params[k].data = torch.from_numpy(v).to(device)
    for (k, v) in params.items():
        if k.endswith('weight'):
            torch.nn.init.xavier_uniform(v)
        if k.endswith('bias'):
            v.data = 0.5*torch.ones_like(v.data).to(device)
    optimizer = optim.Adam(params.values(), lr=learning_rate)

    #prior = np.array([(64*64-2)/(64*64.), 1/(64*64.), 1/(64*64.)]).astype(np.float32)
    #prior = np.array([1/(64*64.), (64*64-2)/(64*64.), 1/(64*64.)]).astype(np.float32)
    #prior = np.array([(64*64-3)/(64*64.), 1/(64*64.), 1/(64*64.), 1/(64*64.)]).astype(np.float32)
    #prior = np.reshape(prior, [1, -1, 1, 1])
    #prior = torch.tensor(np.tile(prior, [1, 1, 64, 64]), device=device)

    logdir = 'ae_relu_kl_0_' + time.strftime("%d-%m-%Y_%H-%M")
    n_validation_samples = 5
    eps = 1e-20
    enc.train()
    dec.train()
    model_forward = lambda ims_tensor: ae_forward(enc, dec, ims_tensor)
    for epoch in range(10): #10 #30
        for (train_ind, rollout) in tqdm(enumerate(train_dataloader)):
            if train_ind >= 50:
                break
            rollout = rollout.to(device)
            ims_tensor = rollout.reshape(-1, 3, 64, 64)
            latent, samples, reconstr = model_forward(ims_tensor)

            optimizer.zero_grad()
            sampled_beta = torch.mean(samples)
            kl_loss = torch.mean(torch.log(sampled_beta) + (1/64*64)/sampled_beta)
            # kl_loss = sampled_beta
            reconstr_loss = torch.mean(
                (ims_tensor - reconstr)**2
            )
            #kl_weight = epoch / 300.
            kl_weight = 0
            loss = kl_weight*kl_loss + reconstr_weight*reconstr_loss
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(epoch, kl_weight*kl_loss.detach().cpu().numpy(), reconstr_weight*reconstr_loss.detach().cpu().numpy())
            validate_model(logdir, epoch, val_dataloader, n_validation_samples, model_forward, params, device)

    print(epoch, kl_weight*kl_loss.detach().cpu().numpy(), reconstr_weight*reconstr_loss.detach().cpu().numpy())
    validate_model(logdir, epoch, val_dataloader, n_validation_samples, model_forward, params, device)

class ObjDataset(Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        rollout = self.data[idx]
        tensor_rollout = []
        for im in rollout:
            im = self._preprocess_im(im)
            im = im.transpose([2, 0, 1])
            im = torch.Tensor(im, device=self.device)
            tensor_rollout.append(im)
        tensor_rollout = torch.stack(tensor_rollout, dim=0)
        return tensor_rollout

    def _preprocess_im(self, im):
        new_im = np.zeros((64, 64, 3), dtype=np.float32)
        group1_locs = np.where(im == 1)
        new_im[group1_locs[0], group1_locs[1], np.ones_like(group1_locs[2])] = 1.
        group2_locs = np.where(im == 2)
        new_im[group2_locs[0], group2_locs[1], 2*np.ones_like(group2_locs[2])] = 1.
        return new_im

def ae_forward(enc, dec, ims_tensor):
    latent = 1 - torch.exp(-enc(ims_tensor))
    samples = gumbel_softmax_sample(
        logits=latent.permute(0, 2, 3, 1),
        temperature=0.1,
    ).permute(0, 3, 1, 2)
    reconstr = dec(samples)
    # reconstr = dec(latent)
    return latent, samples, reconstr

def validate_model(logdir, epoch, val_dataloader, n_validation_samples, model_forward, params, device):
    try:
        os.system('mkdir data/{}'.format(logdir))
        os.system('mkdir data/{}/{}'.format(logdir, epoch))
    except Exception as e:
        pass

    params = {k: v.data.cpu().numpy() for (k,v) in params.items()}
    pickle.dump(params, open('data/{}/{}/params.pkl'.format(logdir, epoch), 'wb'))

    for (val_ind, rollout) in enumerate(val_dataloader):
        if val_ind >= 5:
            break
        # ims = np.tile(val_data[0,val_ind:val_ind+1,:,:,0] - 0.5, [3,1,1,1]).transpose([1,0,2,3]).astype(np.float32)
        rollout = rollout.to(device)
        ims_tensor = rollout.reshape(-1, 3, 64, 64)
        latent, samples, reconstr = model_forward(ims_tensor)

        latent_im = (latent.detach().cpu().numpy()[0, :]).transpose([1, 2, 0])
        latent_im = np.concatenate((np.zeros_like(latent_im[:,:,:1]), latent_im), axis=2)
        samples_im = (samples.detach().cpu().numpy()[0, :]).transpose([1, 2, 0])
        samples_im = np.concatenate((np.zeros_like(samples_im[:,:,:1]), samples_im), axis=2)
        input_im = (ims_tensor.detach().cpu().numpy()[0, :]).transpose([1, 2, 0])
        reconstr_im = (reconstr.detach().cpu().numpy()[0, :]).transpose([1, 2, 0])
        pickle.dump(latent_im, open('data/{}/{}/latent_{}.pkl'.format(logdir, epoch, val_ind), 'wb'))
        pickle.dump(samples_im, open('data/{}/{}/sampled_{}.pkl'.format(logdir, epoch, val_ind), 'wb'))
        pickle.dump(reconstr_im, open('data/{}/{}/reconstr_{}.pkl'.format(logdir, epoch, val_ind), 'wb'))
        pickle.dump(input_im, open('data/{}/{}/im_{}.pkl'.format(logdir, epoch, val_ind), 'wb'))
        cv2.imwrite('data/{}/{}/latent_{}.png'.format(logdir, epoch, val_ind), (255*np.clip(latent_im, 0, 1)).astype(np.uint8)[:,:,-3:])
        cv2.imwrite('data/{}/{}/sampled_{}.png'.format(logdir, epoch, val_ind), (255*np.clip(samples_im, 0, 1)).astype(np.uint8)[:,:,-3:])
        cv2.imwrite('data/{}/{}/im_{}.png'.format(logdir, epoch, val_ind), (255*np.clip(input_im, 0, 1)).astype(np.uint8))
        cv2.imwrite('data/{}/{}/reconstr_{}.png'.format(logdir, epoch, val_ind), (255*np.clip(reconstr_im, 0, 1)).astype(np.uint8))

if __name__ == '__main__':
    test_autoencoder()