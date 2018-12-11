import numpy as np
import cv2
import pickle
import torch
import itertools

# max_obj_size in latent space
def threshold_latent(latent, prev_latent=None, max_obj_size=3):
    if prev_latent is None:
        locs = []
        for channel in latent:
            channel_locs = torch.nonzero(channel > 0.5)
            sorted_channel_locs = sort_merge_channel_locs(channel_locs, max_obj_size)
            locs.append(sorted_channel_locs)
        return locs
    else:
        locs = []
        for (ch_idx, channel) in enumerate(latent):
            prev_channel_locs = torch.nonzero(prev_latent[ch_idx] > 0.5)
            channel_locs = torch.nonzero(channel > 0.5)
            sorted_channel_locs = sort_merge_channel_locs(channel_locs, max_obj_size)
            corr_channel_locs = []
            for prev_loc in prev_channel_locs:
                best_loc = None
                best_dist = None
                for cand_loc in sorted_channel_locs:
                    if best_dist is None or manhat_dist(prev_loc, cand_loc) < best_dist:
                        best_dist = manhat_dist(prev_loc, cand_loc)
                        best_loc = cand_loc
                corr_channel_locs.append(best_loc)
            locs.append(corr_channel_locs)
        return locs

def sort_merge_channel_locs(channel_locs, max_obj_size=3):
    sorted_channel_locs = []
    idx = 0
    for y in range(latent.shape[1]):
        if idx >= channel_locs.shape[0]:
            break
        xs_for_y = []
        while idx < channel_locs.shape[0] \
        and channel_locs[idx][0] == y:
            xs_for_y.append(channel_locs[idx])
            idx += 1
        if xs_for_y:
            xs_for_y = torch.stack(xs_for_y, dim=0)
            for l1 in xs_for_y:
                if not sorted_channel_locs \
                or min(map(lambda l2: manhat_dist(l1, l2), sorted_channel_locs)) >= max_obj_size:
                    sorted_channel_locs.append(l1)
    return sorted_channel_locs


def manhat_dist(l1, l2):
    return torch.abs(l1[0]-l2[0]) + torch.abs(l1[1]-l2[1])

if __name__ == '__main__':
    latent = np.zeros((10, 10))
    latent[[0,1,5,5,5,0],[0,0,5,6,0,5]] = 1
    latent = np.tile(latent.reshape(1, 10, 10), (5, 1, 1))
    latent = torch.Tensor(latent)
    print(threshold_latent(latent))

    latent = np.zeros((10, 10))
    latent[[0,5,5,0],[0,0,5,5]] = 1
    latent = np.tile(latent.reshape(1, 10, 10), (5, 1, 1))
    latent = torch.Tensor(latent)
    prev_latent = np.zeros((10, 10))
    prev_latent[[1,6,4,1],[1,1,4,6]] = 1
    prev_latent = np.tile(prev_latent.reshape(1, 10, 10), (5, 1, 1))
    prev_latent = torch.Tensor(prev_latent)
    print(threshold_latent(latent, prev_latent))
    print(threshold_latent(prev_latent, latent))
