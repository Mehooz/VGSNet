"""
    This is the main trainer script for point cloud AE/VAE experiments.
    Use scripts/train_ae_pc_chair.sh or scripts/train_vae_pc_chair.sh to run.
    Before that, you need to run scripts/pretrain_part_pc_ae_chair.sh or scripts/pretrain_part_pc_vae_chair.sh
    to pretrain part geometry AE/VAE.
"""

import os
import cv2
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from config import add_train_vae_args
from data import PartNetDataset, Tree
import utils
from image_encoder import ImageEncoder

# Use 1-4 CPU threads to train.
# Don't use too many CPU threads, which will slow down the training.
# torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
        fn = os.path.join(self.root, self.object_names[index]+'.npy')
        data = np.load(fn)
        fn_img = os.path.join(self.root.replace('_z_new','_img_3000'), self.object_names[index] + '.npz')
        img_set = np.load(fn_img)['image']
        # idx = np.random.randint(min(data.shape[0],img_set.shape[0]))
        idx = np.random.randint(data.shape[0])

        z = torch.tensor(data, dtype=torch.float32)
        img = img_set[idx,:,:,:]
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img,dtype = torch.float32)

        return (z, img ,self.object_names[index], idx)

    def __len__(self):
        return len(self.object_names)

def train(conf):
    # load network model
    #conf.model_version = 'model_pc_img'
    conf.exp_name = 'pc_ae_chair_3000_test'
    conf.data_path = '/home/zhangxc/tmp/structurenet/data/partnetdata/chair_z_new'
    conf.train_dataset = 'train_no_other_less_than_10_parts.txt'
    conf.val_dataset = 'val_no_other_less_than_10_parts.txt'
    conf.epochs = 200
    conf.part_pc_exp_name = 'image_encoder_st'
    conf.part_pc_model_epoch = 194
    conf.batch_size = 64
    conf.num_point = 3000
    conf.checkpoint_interval = 54


    #models = utils.get_model_module(conf.model_version)

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

    # save config
    torch.save(conf, os.path.join(conf.model_path, conf.exp_name, 'conf.pth'))

    # create models
    encoder = ImageEncoder(256).to(device)
    models=[encoder]
    # create optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=conf.lr)
    model_names=['ImageEncoder']
    optimizers = [encoder_opt]
    optimizer_names = ['encoder']

    # learning rate scheduler
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_opt, \
            step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create training and validation datasets and data loaders
    data_features = ['object']

    train_dataset = PartNetGeoDataset(conf.data_path, conf.train_dataset,True)
    valdt_dataset = PartNetGeoDataset(conf.data_path, conf.val_dataset, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, \
            shuffle=True, collate_fn=utils.collate_feats)
    valdt_dataloader = torch.utils.data.DataLoader(valdt_dataset, batch_size=conf.batch_size, \
            shuffle=True, collate_fn=utils.collate_feats)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)      LR       LatentLoss    GeoLoss   CenterLoss   ScaleLoss   StructLoss  EdgeExists  KLDivLoss   SymLoss   AdjLoss   TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'train'))
        valdt_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'val'))

    # send parameters to device

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

            # make sure the models are in eval mode to deactivate BatchNorm for PartEncoder and PartDecoder
            # there are no other BatchNorm / Dropout in the rest of the network
            encoder.eval()

            # forward pass (including logging)
            total_loss = forward(
                batch=batch, data_features=data_features, encoder=encoder, device=device, conf=conf,
                is_valdt=False, step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time,
                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer,
                lr=encoder_opt.param_groups[0]['lr'], flog=flog)

            # optimize one step
            encoder_scheduler.step()
            encoder_opt.zero_grad()
            total_loss.backward()
            encoder_opt.step()

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
                encoder.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    __ = forward(
                        batch=batch, data_features=data_features, encoder=encoder, device=device, conf=conf,
                        is_valdt=True, step=valdt_step, epoch=epoch, batch_ind=valdt_batch_ind, num_batch=valdt_num_batch, start_time=start_time,
                        log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=valdt_writer,
                        lr=encoder_opt.param_groups[0]['lr'], flog=flog)

    # save the final models
    print("Saving final checkpoint ...... ", end='', flush=True)
    flog.write("Saving final checkpoint ...... ")
    utils.save_checkpoint(
        models=models, model_names=model_names, dirname=os.path.join(conf.model_path, conf.exp_name),
        epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    print("DONE")
    flog.write("DONE\n")

    flog.close()

def forward(batch, data_features, encoder,  device, conf,
            is_valdt=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, flog=None):
    zs = batch[0]
    img= batch[1]

    total_loss = 0
    f = nn.MSELoss(reduce=True)

    # process every data in the batch individually
    for i in range(len(zs)):
        zs[i].to(device)

        # encode object to get root code
        root_code=encoder(img[i].permute(2,0,1).unsqueeze(0).to(device)).to(device)
        total_loss+=f(root_code,zs[i].to(device))


    with torch.no_grad():
        # log to console
        if log_console:
            print(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{total_loss.item():>10.2f}''')
            flog.write(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{total_loss.item():>10.2f}\n''')
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)


    return total_loss

if __name__ == '__main__':
    sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

    parser = ArgumentParser()
    parser = add_train_vae_args(parser)
    config = parser.parse_args()
    config.category = 'Chair'

    Tree.load_category_info(config.category)
    train(config)



