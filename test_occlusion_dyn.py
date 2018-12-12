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
from dyn_model import PairwiseInteract
import pickle
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device("cpu")

def disp(objs, dec, name='objs'):
    objs = [np.clip(np.round(k.detach().cpu().numpy()).astype(np.uint8), 0, 63) for k in objs]
    im = np.zeros((1, 8, 64, 64), dtype=np.float32)
    for obj in objs[0]:
        im[0, 3, obj[0], obj[1]] = 1
    for obj in objs[1]:
        im[0, 4, obj[0], obj[1]] = 1
    im = dec(torch.from_numpy(im)).detach().cpu().numpy().reshape(3, 64, 64).transpose([1,2,0])
    #cv2.imshow(name, im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(name, np.clip(200*im, 0, 255).astype(np.uint8))

if __name__ == '__main__':
    n_classes = 2
    dyn_model = PairwiseInteract(n_classes=n_classes)

    params = pickle.load(open('data/occlusion_dyn_model.pkl', 'rb'))

    for (name, param) in dyn_model.named_parameters():
        param.data = torch.from_numpy(params[name])
    dataset = pickle.load(open('data/occlusion_latent.pkl', 'rb'))

    from occlusion_ae import init_weights
    enc = Encoder([3,96,96,8], [0,0,0,8], 8).to(device)
    dec = Decoder([9,96,96,3], 8).to(device)
    params = {}
    for (k, v) in enc.named_parameters():
        params['enc.'+k.replace('__', '.')] = v
    for (k, v) in dec.named_parameters():
        params['dec.'+k.replace('__', '.')] = v
    params = init_weights(
        params, file='occlusion_params.pkl',
        device=device
    )

    for (n, (prev, cur, targ)) in enumerate(dataset[:1000:50]):
        if all([len(k) for k in prev]):
            for t in range(50):
                if t == 0:
                    prev = [k.cpu().type(torch.FloatTensor) for k in prev]
                    cur = [k.cpu().type(torch.FloatTensor) for k in cur]
                    targ = [k.cpu().type(torch.FloatTensor) for k in targ]
                    pred = dyn_model.forward(cur, prev)
                    disp(prev, dec, 'occ_vids/{}im00.png'.format(n))
                    disp(cur, dec, 'occ_vids/{}im01.png'.format(n))
                    disp(pred, dec, 'occ_vids/{}im02.png'.format(n))
                else:
                    prev = cur
                    cur = pred
                    pred = dyn_model.forward(cur, prev)
                    disp(pred, dec, 'occ_vids/{0:}im{1:02d}.png'.format(n, t+2))
