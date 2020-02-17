import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist as np_cdist

def get_loss_proj(pred, gt,device, loss_type='bce', w=1., min_dist_loss=None,
        dist_mat=None):

    if loss_type=='bce':
        print('BCE loss')
        loss_f=nn.BCELoss()
        loss=loss_f(gt,pred)

    if loss_type=='weighted_bce':
        print('BCE loss')
        loss_f=nn.BCEWithLogitsLoss()
        loss=loss_f(gt,pred)
    if loss_type == 'bce_prob':
        epsilon = 1e-8
        loss = -gt*torch.log(pred+epsilon)*w - (1-gt)*torch.log(torch.abs(1-pred-epsilon))

    if min_dist_loss is not None:
        dist_mat += 1
        dist_mat.to(device)

        gt_white=torch.unsqueeze(torch.unsqueeze(gt, 3),3).repeat([1, 1, 1, 64, 64])
        #gt_white = gt_white.repeat([1,1,1,args.grid_h,args.grid_w])
        #gt_white = gt_white.repeat([1, 1, 1,64,64]).to(device)

        pred_white = torch.unsqueeze(torch.unsqueeze(pred, 3), 3).repeat([1, 1, 1, 64, 64])
        #pred_white = pred_white.repeat([1, 1, 1, args.grid_h, args.grid_w])
        #pred_white = pred_white.repeat([1, 1, 1, 64,64]).to(device)

        pred_mask = (pred_white) + ((1. - pred_white)) * 1e6 * torch.ones_like(pred_white)
        dist_masked_inv = gt_white * dist_mat * pred_mask

        gt_white_th = gt_white + (1. - gt_white) * 1e6 * torch.ones_like(gt_white)
        dist_masked = gt_white_th * dist_mat * pred_white

        min_dist = dist_masked.min(4)[0].min(3)[0]
        min_dist_inv = dist_masked_inv.min(4)[0].min(3)[0]

    #return loss, min_dist, min_dist_inv
    return loss


def grid_dist(grid_h, grid_w):
    '''
    Compute distance between every point in grid to every other point
    '''
    x, y = np.meshgrid(range(grid_h), range(grid_w), indexing='ij')
    grid = np.asarray([[x.flatten()[i],y.flatten()[i]] for i in range(len(x.flatten()))])
    grid_dist = np_cdist(grid,grid)
    grid_dist = np.reshape(grid_dist, [grid_h, grid_w, grid_h, grid_w])
    return torch.from_numpy(grid_dist).float()

