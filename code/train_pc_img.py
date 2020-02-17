"""
    This is the main trainer script for point cloud AE/VAE experiments.
    Use scripts/train_ae_pc_chair.sh or scripts/train_vae_pc_chair.sh to run.
    Before that, you need to run scripts/pretrain_part_pc_ae_chair.sh or scripts/pretrain_part_pc_vae_chair.sh
    to pretrain part geometry AE/VAE.
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
from data import PartNetDataset, Tree
import utils

# Use 1-4 CPU threads to train.
# Don't use too many CPU threads, which will slow down the training.
# torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(conf):
    # load network model
    conf.model_version = 'model_pc_img'
    conf.exp_name = 'pc_ae_table_3000_test'
    conf.data_path = '/home/zhangxc/tmp/structurenet/data/partnetdata/table_hier_3000'
    conf.train_dataset = 'train_no_other_less_than_10_parts.txt'
    conf.val_dataset = 'val_no_other_less_than_10_parts.txt'
    conf.epochs = 200
    conf.part_pc_exp_name = 'part_pc_ae_table'
    conf.part_pc_model_epoch = 194
    conf.batch_size = 64
    conf.num_point = 3000
    conf.checkpoint_interval = 54

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

    # save config
    torch.save(conf, os.path.join(conf.model_path, conf.exp_name, 'conf.pth'))

    # create models
    encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=not conf.non_variational)
    decoder = models.RecursiveDecoder(conf)
    models = [encoder, decoder]
    model_names = ['encoder', 'decoder']

    '''
    # load pretrained part AE/VAE
    pretrain_ckpt_dir = os.path.join(conf.model_path, conf.part_pc_exp_name)
    print(pretrain_ckpt_dir)
    pretrain_ckpt_epoch = conf.part_pc_model_epoch
    print(f'Loading ckpt from {pretrain_ckpt_dir}: epoch {pretrain_ckpt_epoch}')
    __ = utils.load_checkpoint(
        models=[encoder.node_encoder.part_encoder, decoder.node_decoder.part_decoder],
        model_names=['part_pc_encoder', 'part_pc_decoder'],
        dirname=pretrain_ckpt_dir,
        epoch=pretrain_ckpt_epoch,
        strict=True)

    # set part_encoder and part_decoder BatchNorm to eval mode
    encoder.node_encoder.part_encoder.eval()
    for param in encoder.node_encoder.part_encoder.parameters():
        param.requires_grad = False
    decoder.node_decoder.part_decoder.eval()
    for param in decoder.node_decoder.part_decoder.parameters():
        param.requires_grad = False
    '''
    # create optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=conf.lr)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=conf.lr)
    optimizers = [encoder_opt, decoder_opt]
    optimizer_names = ['encoder', 'decoder']

    # learning rate scheduler
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_opt, \
            step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_opt, \
            step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create training and validation datasets and data loaders
    data_features = ['object']
    train_dataset = PartNetDataset(conf.data_path, conf.train_dataset, data_features, \
            load_geo=conf.load_geo)
    valdt_dataset = PartNetDataset(conf.data_path, conf.val_dataset, data_features, \
            load_geo=conf.load_geo)
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

            # make sure the models are in eval mode to deactivate BatchNorm for PartEncoder and PartDecoder
            # there are no other BatchNorm / Dropout in the rest of the network
            for m in models:
                m.eval()

            # forward pass (including logging)
            total_loss = forward(
                batch=batch, data_features=data_features, encoder=encoder, decoder=decoder, device=device, conf=conf,
                is_valdt=False, step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time,
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
                        batch=batch, data_features=data_features, encoder=encoder, decoder=decoder, device=device, conf=conf,
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

def forward(batch, data_features, encoder, decoder, device, conf,
            is_valdt=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, flog=None):
    objects = batch[data_features.index('object')]

    losses = {
        'latent': torch.zeros(1, device=device),
        'geo': torch.zeros(1, device=device),
        'center': torch.zeros(1, device=device),
        'scale': torch.zeros(1, device=device),
        'leaf': torch.zeros(1, device=device),
        'exists': torch.zeros(1, device=device),
        'semantic': torch.zeros(1, device=device),
        'edge_exists': torch.zeros(1, device=device),
        'kldiv': torch.zeros(1, device=device),
        'sym': torch.zeros(1, device=device),
        'adj': torch.zeros(1, device=device)}

    # process every data in the batch individually
    for obj in objects:
        obj.to(device)

        # encode object to get root code
        root_code = encoder.encode_structure(obj=obj)

        # get kldiv loss
        if not conf.non_variational:
            root_code, obj_kldiv_loss = torch.chunk(root_code, 2, 1)
            obj_kldiv_loss = -obj_kldiv_loss.sum() # negative kldiv, sum over feature dimensions
            losses['kldiv'] = losses['kldiv'] + obj_kldiv_loss

        # decode root code to get reconstruction loss
        img_fea = obj.root.image.to(device)
        # print(img_fea.shape)

        obj_losses = decoder.structure_recon_loss(z=root_code, gt_tree=obj,img_fea=img_fea)
        for loss_name, loss in obj_losses.items():
            losses[loss_name] = losses[loss_name] + loss

    for loss_name in losses.keys():
        losses[loss_name] = losses[loss_name] / len(objects)

    losses['latent'] *= conf.loss_weight_latent
    losses['geo'] *= conf.loss_weight_geo
    losses['center'] *= conf.loss_weight_center
    losses['scale'] *= conf.loss_weight_scale
    losses['leaf'] *= conf.loss_weight_leaf
    losses['exists'] *= conf.loss_weight_exists
    losses['semantic'] *= conf.loss_weight_semantic
    losses['edge_exists'] *= conf.loss_weight_edge_exists
    losses['kldiv'] *= conf.loss_weight_kldiv
    losses['sym'] *= conf.loss_weight_sym
    losses['adj'] *= conf.loss_weight_adj

    total_loss = 0
    for loss in losses.values():
        total_loss += loss

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
                f'''{losses['latent'].item():>11.2f} '''
                f'''{losses['geo'].item():>11.2f} '''
                f'''{losses['center'].item():>11.2f} '''
                f'''{losses['scale'].item():>11.2f} '''
                f'''{(losses['leaf']+losses['exists']+losses['semantic']).item():>11.2f} '''
                f'''{losses['edge_exists'].item():>11.2f} '''
                f'''{losses['kldiv'].item():>10.2f} '''
                f'''{losses['sym'].item():>10.2f} '''
                f'''{losses['adj'].item():>10.2f} '''
                f'''{total_loss.item():>10.2f}''')
            flog.write(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{losses['latent'].item():>11.2f} '''
                f'''{losses['geo'].item():>11.2f} '''
                f'''{losses['center'].item():>11.2f} '''
                f'''{losses['scale'].item():>11.2f} '''
                f'''{(losses['leaf']+losses['exists']+losses['semantic']).item():>11.2f} '''
                f'''{losses['edge_exists'].item():>11.2f} '''
                f'''{losses['kldiv'].item():>10.2f} '''
                f'''{losses['sym'].item():>10.2f} '''
                f'''{losses['adj'].item():>10.2f} '''
                f'''{total_loss.item():>10.2f}\n''')
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)
            tb_writer.add_scalar('latent_loss', losses['latent'].item(), step)
            tb_writer.add_scalar('geo_loss', losses['geo'].item(), step)
            tb_writer.add_scalar('center_loss', losses['center'].item(), step)
            tb_writer.add_scalar('scale_loss', losses['scale'].item(), step)
            tb_writer.add_scalar('leaf_loss', losses['leaf'].item(), step)
            tb_writer.add_scalar('exists_loss', losses['exists'].item(), step)
            tb_writer.add_scalar('semantic_loss', losses['semantic'].item(), step)
            tb_writer.add_scalar('edge_exists_loss', losses['edge_exists'].item(), step)
            tb_writer.add_scalar('kldiv_loss', losses['kldiv'].item(), step)
            tb_writer.add_scalar('sym_loss', losses['sym'].item(), step)
            tb_writer.add_scalar('adj_loss', losses['adj'].item(), step)

    return total_loss

if __name__ == '__main__':
    sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

    parser = ArgumentParser()
    parser = add_train_vae_args(parser)
    config = parser.parse_args()
    config.category = 'Table'

    Tree.load_category_info(config.category)
    train(config)



