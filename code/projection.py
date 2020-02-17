import torch
import numpy as np

def cont_proj(pcl, grid_h, grid_w,device ,sigma_sq=0.5):
    '''
    Continuous approximation of Orthographic projection of point cloud
    to obtain Silhouette
    args:
            pcl: float, (N_batch,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            grid_h, grid_w: int, ();
                     output depth map height and width
    returns:
            grid_val: float, (N_batch,H,W);
                      output silhouette
    '''
    xyz=torch.split(pcl,1,dim=2)
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]


    pcl_norm = torch.cat([x, y, z], 2)
    pcl_xy = torch.cat([x,y], 2).to(device)
    out_grid = torch.meshgrid(torch.range(0,grid_h-1), torch.range(0,grid_w-1))
    out_grid = [out_grid[0].float(), out_grid[1].float()]
    grid_z = torch.unsqueeze(torch.zeros_like(out_grid[0]), 2) # (H,W,1)
    grid_xyz = torch.cat([torch.stack(out_grid,2), grid_z],2)  # (H,W,3)
    grid_xy = torch.stack(out_grid,2).to(device)               # (H,W,2)
    grid_diff = torch.unsqueeze(torch.unsqueeze(pcl_xy, 2), 2).to(device) - grid_xy # (BS,N_PTS,H,W,2)
    grid_val = apply_kernel(grid_diff, sigma_sq)    # (BS,N_PTS,H,W,2)
    grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]  # (BS,N_PTS,H,W)
    grid_val = torch.sum(grid_val,1)   # (BS,H,W)
    th=torch.nn.Tanh()
    grid_val = th(grid_val)
    return grid_val


def disc_proj(pcl, grid_h, grid_w):
    '''
    Discrete Orthographic projection of point cloud
    to obtain Silhouette
    Handles only batch size 1 for now
    args:
            pcl: float, (N_batch,N_Pts,3); input point cloud
                     values assumed to be in (-1,1)
            grid_h, grid_w: int, ();
                     output depth map height and width
    returns:
            grid_val: float, (N_batch,H,W); output silhouette
    '''
    xyz = torch.split(pcl, 1, dim=2)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]


    pcl_norm = torch.cat([x, y, z], 2)
    pcl_xy = torch.cat([x,y], 2)
    xy_indices = pcl_xy.long()
    #TODO
    xy_values = torch.ones_like(xy_indices[:,:,0]).float()
    out_grid = torch.zeros(grid_h,grid_w).scatter(0, xy_indices[0],xy_values)
    out_grid = torch.unsqueeze(out_grid,0)
    return out_grid


def apply_kernel(x, sigma_sq=0.5):
    '''
    Get the un-normalized gaussian kernel with point co-ordinates as mean and
    variance sigma_sq
    args:
            x: float, (BS,N_PTS,H,W,2); mean subtracted grid input
            sigma_sq: float, (); variance of gaussian kernel
    returns:
            out: float, (BS,N_PTS,H,W,2); gaussian kernel
    '''
    out = (torch.exp(-(x**2)/(2.*sigma_sq)))
    return out


def perspective_transform(xyz, device,batch_size):
    '''
    Perspective transform of pcl; Intrinsic camera parameters are assumed to be
    known (here, obtained using parameters of GT image renderer, i.e. Blender)
    Here, output grid size is assumed to be (64,64) in the K matrix
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
    returns:
            xyz_out: float, (BS,N_PTS,3); perspective transformed point cloud
    '''
    K = np.array([
            [120., 0., -32.],
            [0., 120., -32.],
            [0., 0., 1.]]).astype(np.float32)
    K = np.expand_dims(K, 0)
    K = np.tile(K, [batch_size,1,1])

    xyz_out = torch.matmul(torch.from_numpy(K).float().to(device), xyz.permute(0,2,1))
    xy_out = xyz_out[:,:2]/abs(torch.unsqueeze(xyz[:,:,2],1))
    xyz_out = torch.cat([xy_out, abs(xyz_out[:,2:])],1)
    return xyz_out.permute(0,2,1)


def world2cam(xyz, az, el, batch_size,device, N_PTS=1024):
    '''
    Convert pcl from world co-ordinates to camera co-ordinates
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            az: float, (BS); azimuthal angle of camera in radians
            elevation: float, (BS); elevation of camera in radians
            batch_size: int, (); batch size
            N_PTS: float, (); number of points in point cloud
    returns:
            xyz_out: float, (BS,N_PTS,3); output point cloud in camera
                        co-ordinates
    '''
    # Distance of object from camera - fixed to 2
    d = 2.
    # Calculate translation params
    # Camera origin calculation - az,el,d to 3D co-ord
    tx, ty, tz = [0, 0, d]
    rotmat_az =torch.stack(
        ((torch.stack((torch.ones_like(az), torch.zeros_like(az), torch.zeros_like(az)),0),
        torch.stack((torch.zeros_like(az), torch.cos(az), -torch.sin(az)),0),
        torch.stack((torch.zeros_like(az), torch.sin(az), torch.cos(az)),0)))
    ,0).to(device)

    rotmat_el = torch.stack(
        ((torch.stack((torch.cos(el), torch.zeros_like(az), torch.sin(el)),0),
        torch.stack((torch.zeros_like(az), torch.ones_like(az), torch.zeros_like(az)),0),
        torch.stack((-torch.sin(el), torch.zeros_like(az), torch.cos(el)),0)))
    ,0).to(device)

    rotmat_az = rotmat_az.permute(2,0,1)
    rotmat_el = rotmat_el.permute(2,0,1)
    rotmat = torch.matmul(rotmat_el, rotmat_az)

    t=torch.Tensor([tx,ty,tz])

    tr_mat = torch.unsqueeze(t,0).repeat([batch_size,1]) # [B,3]
    tr_mat = torch.unsqueeze(tr_mat,2) # [B,3,1]
    tr_mat = tr_mat.permute(0,2,1) # [B,1,3]
    tr_mat = tr_mat.repeat([1,N_PTS,1]).to(device) # [B,1024,3]

    xyz_out = torch.matmul(rotmat,xyz.permute(0,2,1)- tr_mat.permute(0,2,1))

    return xyz_out.permute(0,2,1)

if __name__ == '__main__':
    print(cont_proj(torch.rand(8,3000,3),64,64,'cpu'))