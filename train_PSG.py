import config
from model.PSG2 import PSGNet2
from torch.utils.data import DataLoader
from data_fetcher import *
from torch import nn
from torch import optim
from model.cd_loss import ChamferLoss
from model.emd_loss import EMD
from utils import meter
from scipy.stats import wasserstein_distance
from collections import OrderedDict
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


def train(train_loader, net, criterion, optimizer, epoch):
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    losses = meter.AverageValueMeter()
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    net.train()

    for i, (views,pcs) in enumerate(train_loader):
        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        preds,_ = net(views)

        loss = criterion(preds, pcs)

        # for i in range(0,preds.cpu().detach().numpy().shape[0]):
        #     wasserstein_distance(np.reshape(preds.cpu().detach().numpy()[i],(-1)),
        #                           np.reshape(preds.cpu().detach().numpy()[i],(-1)))

        # prec.add(preds.detach())
        losses.add(loss.item())  # batchsize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.print_frequncy == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Loss {losses.value()[0]:.4f} \t')
                  # f'Prec@1 {prec.value(1):.3f}\t')

    # print(f'prec at epoch {epoch}: {prec.value(1)} ')
    print(f'Loss at epoch{epoch}:{losses.value()[0]}')
    return  losses.value()[0]

def save_record(epoch, prec1, net: nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict, os.path.join(config.psg_net.ckpt_record_folder, f'epoch{epoch}_{prec1:.4f}.pth'))

def save_ckpt(epoch, net, optimizer_all, training_conf=config.psg_net):
    ckpt = dict(
        epoch=epoch,
        model=net.module.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
        training_conf=training_conf
    )
    torch.save(ckpt, config.psg_net.ckpt_file)

def main():
    print('Training PSG')
    train_dataset = PSGPointViewFetcher('./data_train/',status='train',data_sample=True)
    val_dataset = PSGPointViewFetcher('./data_test/',status='test',data_sample=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=16,
                              num_workers=config.num_workers,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                             batch_size=16,
                             num_workers=config.num_workers,
                             shuffle=True,
                             drop_last=True)

    resume_epoch = 0

    net = PSGNet2()
    net = net.to(device=config.device)
    net = nn.DataParallel(net)
    fc_param = [{'params': v} for k, v in net.named_parameters()]

    if config.psg_net.train.optim == 'Adam':
        optimizer_all = optim.Adam(net.parameters(), config.psg_net.train.all_lr,
                                   weight_decay=config.psg_net.train.weight_decay)
    elif config.psg_net.train.optim == 'SGD':
        optimizer_all = optim.SGD(net.parameters(), config.psg_net.train.all_lr,
                                  momentum=config.psg_net.train.momentum,
                                  weight_decay=config.psg_net.train.weight_decay)
    else:
        raise NotImplementedError
    print(f'use {config.psg_net.train.optim} optimizer')

    if config.psg_net.train.resume:
        print(f'loading pretrained model from {config.psg_net.ckpt_file}')
        checkpoint = torch.load(config.psg_net.ckpt_file)
        state_dict = checkpoint['model']

        net.module.load_state_dict(checkpoint['model'])
        optimizer_all.load_state_dict(checkpoint['optimizer_all'])

        if config.psg_net.train.resume_epoch is not None:
            resume_epoch = config.psg_net.train.resume_epoch
        else:
            resume_epoch = max(checkpoint['epoch_pc'], checkpoint['epoch_all'])

    if config.psg_net.train.iter_train == False:
        print('No iter')
        lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, 5, 0.3)
    else:
        print('iter')
        lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, 6, 0.3)

    criterion = ChamferLoss()
    # criterion = EMD()
    # criterion = criterion.to(device=config.device)
    ck_dict = torch.load('./epoch0.pth')

    ck_dict_new = OrderedDict()
    for key in ck_dict:
        ck_dict_new['module.'+key] = ck_dict[key]

    net.load_state_dict(ck_dict_new)
    for epoch in range(resume_epoch, config.pv_net.train.max_epoch):
        losses = train(train_loader, net, criterion, optimizer_all, epoch)

        # with torch.no_grad():
        #     prec1, retrieval_map = validate(val_loader, net, epoch)

        save_ckpt(epoch, net, optimizer_all)
        save_record(epoch, losses , net.module)

    print('Train Finished!')


if __name__ == '__main__':
    main()

