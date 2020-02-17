"""
    This is the trainer script for pretraining part point cloud AE/VAE for StructureNet point cloud experiments.
    Use scripts/pretrain_part_pc_ae_chair.sh or scripts/pretrain_part_pc_vae_chair.sh to run.
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
from config import add_train_vae_args
import utils
import cv2
from tensorboardX import SummaryWriter
# Use 1-4 CPU threads to train.
# Don't use too many CPU threads, which will slow down the training.
torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

"""
    For each shape, randomly sample a part to feed to the network.
    If use_local_frame=True, we re-center and re-scale the part into
    a unit sphere before feeding to the network.
"""
def normal_pc(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc/pc_L_max
    return pc

class PartNetGeoDataset(torch.utils.data.Dataset):
    def __init__(self, root, object_list, use_local_frame):
        self.root = root
        self.use_local_frame = use_local_frame

        if isinstance(object_list, str):
            with open(os.path.join(root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]

        else:
            self.object_names = object_list

    def __getitem__(self, index):
        theta_x = -np.pi / 6
        theta_y = np.pi / 4
        Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        fn = os.path.join(self.root, self.object_names[index]+'.npz')
        data = np.load(fn)['parts']
        fn_img = os.path.join(self.root.replace('_geo','_img'), self.object_names[index] + '.npz')
        img_set = np.load(fn_img)['image']
        # idx = np.random.randint(min(data.shape[0],img_set.shape[0]))
        idx = np.random.randint(data.shape[0])
        pts = data[idx, :, :]
        pts = normal_pc(pts)
        pts = np.dot(pts, Ry)
        pts = np.dot(pts, Rx)
        pts = torch.tensor(pts, dtype=torch.float32)
        img = img_set[idx,:,:,:]
        # img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img,dtype = torch.float32)
        if self.use_local_frame:
            pts = pts - pts.mean(dim=0)
            pts = pts / pts.pow(2).sum(dim=1).max().sqrt()
        return (pts, img ,self.object_names[index], idx)

    def __len__(self):
        return len(self.object_names)

# def weights_init(m):
#
#     classname=m.__class__.__name__
#     if classname.find('ConvTranspose2d') != -1:
#         torch.nn.init.uniform_(m.weight.data,a = -1.0,b =1.0)
#         torch.nn.init.uniform_(m.bias.data,a = -1.0,b =1.0)
#     if classname.find('Linear') != -1:
#         torch.nn.init.uniform_(m.weight.data,a = -1.0,b =1.0)
#         torch.nn.init.uniform_(m.bias.data,a = -1.0,b =1.0)

def train(conf):
    # load network model
    # os.environ['CUDA_VISIBLE_DEVICES']='2'
    conf.exp_name = 'part_pc_ae_table_3000_vst_fea_kl0.01'
    conf.category = 'Table'
    conf.data_path = '../data/partnetdata/table_geo_3000'
    conf.train_dataset = 'train_no_other_less_than_10_parts.txt'
    conf.val_dataset = 'val_no_other_less_than_10_parts.txt'
    conf.epochs = 300
    conf.model_version = 'model_part_vst_fea'
    conf.batch_size = 1
    # conf.lr = 1e-3
    conf.lr = 1e-4
    conf.lr_decay_every = 5000
    conf.lr_decay_by = 0.9
    conf.device = 'cuda:0'
    conf.num_point = 3000
    conf.non_variational = False
    conf.checkpoint_interval = 10000
    conf.loss_weight_kldiv = 0.01



    models = utils.get_model_module(conf.model_version)

    # check if training run already exists. If so, delete it.
    if os.path.exists(os.path.join(conf.log_path, conf.exp_name)) or \
       os.path.exists(os.path.join(conf.model_path, conf.exp_name)):
        response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
        if response != 'y':
            sys.exit()
    if os.path.exists(os.path.join(conf.log_path, conf.exp_name)):
        shutil.rmtree(os.path.join(conf.log_path, conf.exp_name))
    if os.path.exists(os.path.join(conf.model_path, conf.exp_name)):
        shutil.rmtree(os.path.join(conf.model_path, conf.exp_name))

    # create directories for this run
    os.makedirs(os.path.join(conf.model_path, conf.exp_name))
    os.makedirs(os.path.join(conf.log_path, conf.exp_name))

    # file log
    flog = open(os.path.join(conf.log_path, conf.exp_name, 'train.log'), 'w')

    # set training device
    device = torch.device(conf.device)
    print(f'Using device: {conf.device}')
    flog.write(f'Using device: {conf.device}\n')

    # log the object category information
    print(f'Object Category: {conf.category}')
    flog.write(f'Object Category: {conf.category}\n')

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (conf.seed))
    flog.write(f'Random Seed: {conf.seed}\n')
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # create models
    encoder = models.PartEncoder(feat_len=256, probabilistic=not conf.non_variational)
    # encoder = models.PartEncoder(feat_len=conf.geo_feat_size)
    decoder = models.PartImgDecoder(feat_len=256, num_point=conf.num_point)
    # decoder.apply(weights_init)
    # decoder.load_state_dict(
    #     torch.load('/home/zhangxc/tmp/structurenet/data/models/part_pc_ae_bed_3000_img/212_net_part_pc_decoder.pth'))
    models = [encoder, decoder]
    model_names = ['part_pc_encoder', 'part_pc_decoder']

    # create optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    optimizers = [encoder_opt, decoder_opt]
    optimizer_names = ['part_pc_encoder', 'part_pc_decoder']

    # learning rate scheduler
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_opt, \
            step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_opt, \
            step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create training and validation datasets and data loaders
    train_dataset = PartNetGeoDataset(conf.data_path, conf.train_dataset, use_local_frame=conf.use_local_frame)
    valdt_dataset = PartNetGeoDataset(conf.data_path, conf.val_dataset, use_local_frame=conf.use_local_frame)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, \
                                                   shuffle=True, collate_fn=utils.collate_feats, num_workers=4)
    valdt_dataloader = torch.utils.data.DataLoader(valdt_dataset, batch_size=conf.batch_size, \
                                                   shuffle=True, collate_fn=utils.collate_feats, num_workers=4)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch    Dataset    Iteration    Progress(%)     LR        CD        MSE        Project     KLDivLoss     TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'train'))
        valdt_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'val'))

    # save config
    torch.save(conf, os.path.join(conf.model_path, conf.exp_name, 'conf.pth'))

    # send parameters to device
    for m in models:
        m.to(device)
    for o in optimizers:
        utils.optimizer_to_device(o, device)

    # start training
    print("Starting training ...... ")
    flog.write('Starting training ......\n')

    start_time = time.time()

    last_checkpoint_step = None
    last_train_console_log_step, last_valdt_console_log_step = None, None
    train_num_batch, valdt_num_batch = len(train_dataloader), len(valdt_dataloader)

    # train for every epoch
    for epoch in range(conf.epochs):
        if not conf.no_console_log:
            print(f'training run {conf.exp_name}')
            flog.write(f'training run {conf.exp_name}\n')
            print(header)
            flog.write(header+'\n')

        train_batches = enumerate(train_dataloader, 0)
        valdt_batches = enumerate(valdt_dataloader, 0)

        train_fraction_done, valdt_fraction_done = 0.0, 0.0
        valdt_batch_ind = -1

        # train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            total_loss = forward(
                batch=batch, encoder=encoder, decoder=decoder, device=device, conf=conf,
                is_valdt=False, step=train_step, epoch=epoch, batch_ind=train_batch_ind,
                num_batch=train_num_batch, start_time=start_time,
                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer,
                lr=encoder_opt.param_groups[0]['lr'], flog=flog)

            # optimize one step
            encoder_scheduler.step()
            decoder_scheduler.step()
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            total_loss.backward()
            encoder_opt.step()
            decoder_opt.step()

            # save checkpoint
            with torch.no_grad():
                if last_checkpoint_step is None or \
                        train_step - last_checkpoint_step >= conf.checkpoint_interval:
                    print("Saving checkpoint ...... ", end='', flush=True)
                    flog.write("Saving checkpoint ...... ")
                    utils.save_checkpoint(
                        models=models, model_names=model_names, dirname=os.path.join(conf.model_path, conf.exp_name),
                        epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names)
                    print("DONE")
                    flog.write("DONE\n")
                    last_checkpoint_step = train_step

            # validate one batch
            while valdt_fraction_done <= train_fraction_done and valdt_batch_ind+1 < valdt_num_batch:
                valdt_batch_ind, batch = next(valdt_batches)

                valdt_fraction_done = (valdt_batch_ind + 1) / valdt_num_batch
                valdt_step = (epoch + valdt_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_valdt_console_log_step is None or \
                        valdt_step - last_valdt_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_valdt_console_log_step = valdt_step

                # set models to evaluation mode
                for m in models:
                    m.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    __ = forward(
                        batch=batch, encoder=encoder, decoder=decoder, device=device, conf=conf,
                        is_valdt=True, step=valdt_step, epoch=epoch, batch_ind=valdt_batch_ind,
                        num_batch=valdt_num_batch, start_time=start_time,
                        log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=valdt_writer,
                        lr=encoder_opt.param_groups[0]['lr'], flog=flog)

    # save the final models
    print("Saving final checkpoint ...... ", end='', flush=True)
    flog.write('Saving final checkpoint ...... ')
    utils.save_checkpoint(
        models=models, model_names=model_names, dirname=os.path.join(conf.model_path, conf.exp_name),
        epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    print("DONE")
    flog.write("DONE\n")

    flog.close()


def forward(batch, encoder, decoder,device, conf,
            is_valdt=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, flog=None):
    pts = torch.cat([item.unsqueeze(dim=0) for item in batch[0]], dim=0).to(device)
    imgs = torch.cat([item.unsqueeze(dim=0) for item in batch[1]], dim=0).to(device)
    imgs = imgs.permute(0,3,1,2)


    net = encoder(pts,imgs)

    if not conf.non_variational:
        img_fea, pc_fea, kldiv_loss = torch.chunk(net, 3, 1)
        kldiv_loss = -kldiv_loss.sum(dim=1).mean()
    else:
        kldiv_loss = net.new_tensor(0)
        img_fea, pc_fea = torch.chunk(net, 2, 1)

    pred_pc, pred_img = decoder(torch.cat((img_fea,pc_fea),1))
    cd_loss, mse_loss, = decoder.loss(pred_pc, pts,pred_img,imgs,4,pred_img.shape[0],conf.device,3000)

    total_loss = cd_loss + mse_loss * 0.1  + kldiv_loss * conf.loss_weight_kldiv
    # total_loss =  kldiv_loss * conf.loss_weight_kldiv

    with torch.no_grad():
        # log to console
        if log_console:
            print(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}% '''
                f'''{lr:>5.2E} '''
                f'''{cd_loss.item():>11.2f} '''
                f'''{mse_loss.item():>11.2f} '''
                # f'''{project_loss.item():>11.2f} '''
                f'''{kldiv_loss.item() if not conf.non_variational else 0:>10.2f} '''
                f'''{total_loss.item():>10.2f}''')
            flog.write(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}% '''
                f'''{lr:>5.2E} '''
                f'''{cd_loss.item():>11.2f} '''
                f'''{mse_loss.item():>11.2f} '''
                # f'''{project_loss.item():>11.2f} '''
                f'''{kldiv_loss.item() if not conf.non_variational else 0:>10.2f} '''
                f'''{total_loss.item():>10.2f}\n''')
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('total loss', total_loss.item(), step)
            tb_writer.add_scalar('cd loss', cd_loss.item(), step)
            tb_writer.add_scalar('mse loss', mse_loss.item(), step)
            # tb_writer.add_scalar('project loss', project_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)
            # tb_writer.add_scalar('recon_loss', recon_loss.item(), step)
            if not conf.non_variational:
                tb_writer.add_scalar('kldiv_loss', kldiv_loss.item(), step)

    return total_loss


if __name__ == '__main__':
    sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

    parser = ArgumentParser()
    parser = add_train_vae_args(parser)
    parser.add_argument('--use_local_frame', action='store_true', default=False, help='factorize out 3-dim center + 1-dim scale')
    config = parser.parse_args()


    train(conf=config)

