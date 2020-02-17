import os
import model
from torchvision import transforms
import os.path
from glob import glob
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import threading
import queue as Queue
from utils.pc_aug import normal_pc,pc_aug_funs

def get_info(shapes_dir, isView=False):
    names_dict = {}
    if isView:
        for shape_dir in shapes_dir:
            name = '_'.join(os.path.split(shape_dir)[1].split('.')[0].split('_')[:-1])
            if name in names_dict:
                names_dict[name].append(shape_dir)
            else:
                names_dict[name] = [shape_dir]
    else:
        for shape_dir in shapes_dir:
            name = os.path.split(shape_dir)[1].split('.')[0]
            names_dict[name] = shape_dir

    return names_dict

class ViewDataFetcher(Dataset):
    def __init__(self,view_root,base_model_name = model.ALEXNET, status = 'train'):
        super(ViewDataFetcher,self).__init__()

        self.status = status
        self.view_list = []
        self.lbl_list = []

        if base_model_name in (model.ALEXNET,model.VGG13,model.VGG13BN,model.VGG11BN,model.VGG16,model.RESNET50):
            self.img_size = 224
        elif base_model_name in (model.RESNET101):
            self.img_size = 227
        elif base_model_name in model.INCEPTION_V3:
            self.img_size = 299
        else:
            raise NotImplementedError

        # tranforms.Compose 将多个transform操作组合在一起，Resize用来改变大小，ToTensor将(H,W,C)变为(C,H,W)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        for class_floder in os.listdir(view_root):
            for file_floder in os.listdir(view_root+class_floder):
                for perspect_floder in os.listdir(view_root+class_floder+'/'+file_floder):
                    self.view_list.append(view_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/view.jpg')
                    self.lbl_list.append(int(class_floder))

        self.view_num = len(self.view_list[0])

        print(f'{status} data num: {len(self.view_list)}')

    def __getitem__(self, idx):
        views = self.transform(Image.open(self.view_list[idx]))
        lbl = self.lbl_list[idx]
        return views.float(), lbl

    def __len__(self):
        return len(self.view_list)

class PointViewDataFetcher(Dataset):
    def __init__(self, data_root, base_model_name=model.VGG16, status='train', pc_input_num=1040):
        super(PointViewDataFetcher, self).__init__()

        self.status = status
        self.view_list = []
        self.pc_list = []
        self.lbl_list = []
        self.pc_input_num = pc_input_num

        if base_model_name in (model.ALEXNET, model.VGG13, model.VGG13BN, model.VGG11BN, model.VGG16):
            self.img_sz = 224
        elif base_model_name in (model.RESNET50, model.RESNET101):
            self.img_sz = 224
        elif base_model_name in model.INCEPTION_V3:
            self.img_sz = 299
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Resize(self.img_sz),
            transforms.ToTensor()
        ])


        for class_floder in os.listdir(data_root):
            for file_floder in os.listdir(data_root+class_floder):
                for perspect_floder in os.listdir(data_root+class_floder+'/'+file_floder):
                    self.view_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/view.jpg')
                    self.lbl_list.append(int(class_floder))
                    self.pc_list.append(data_root + class_floder + '/' + file_floder + '/' + perspect_floder + '/pc.npy')

        self.view_num = len(self.view_list[0])

        print(f'{status} data num: {len(self.view_list)}')

    def __getitem__(self, idx):
        views = self.transform(Image.open(self.view_list[idx]))
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        lbl = self.lbl_list[idx]
        while pc.shape[0]<1040:
            pc = np.repeat(pc,2,axis=0)
            pc = pc[:self.pc_input_num]
        pc = pc/100
        pc = normal_pc(pc)
        pc = np.transpose(pc)

        return views.float(), torch.from_numpy(pc).float(),lbl

    def __len__(self):
        return len(self.pc_list)

class PSGPointViewFetcher(Dataset):
    def __init__(self,data_root,status,pc_input_num=3392,data_sample=False):
        super(PSGPointViewFetcher,self).__init__()
        self.data_root = data_root
        self.view_list = []
        self.pc_list = []
        self.pc_input_num = pc_input_num
        self.status = status

        for class_floder in os.listdir(data_root):
            # print(class_floder)
            if data_sample == True:
                if class_floder != '13':
                    continue

            for file_floder in os.listdir(data_root+class_floder):
                for perspect_floder in os.listdir(data_root+class_floder+'/'+file_floder):
                    self.pc_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/pc.npy')
                    self.view_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/view.jpg')

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.view_list)}')

    def __getitem__(self, idx):
        views = self.transform(Image.open(self.view_list[idx]))
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        while pc.shape[0]<3392:
            pc = np.repeat(pc,2,axis=0)
            pc = pc[:self.pc_input_num]
        pc = pc[:self.pc_input_num]
        pc = pc/100
        pc = normal_pc(pc)
        # pc = pc.transpose()
        # pc = np.expand_dims(pc.transpose(), axis=2)

        return views.float(), torch.from_numpy(pc).float()

    def __len__(self):
        return len(self.pc_list)

class GeneratorFetcher(Dataset):
    def __init__(self,data_root,status,pc_input_num=1040,data_sample=False):
        super(GeneratorFetcher,self).__init__()
        self.data_root = data_root
        self.view_list = []
        self.pc_list = []
        self.pc_part_list = []
        self.pc_input_num = pc_input_num
        self.status = status

        for class_floder in os.listdir(data_root):
            # print(class_floder)
            if data_sample == True:
                if class_floder not in ['12','0','1', '2' ,'6', '8'] :
                    continue

            for file_floder in os.listdir(data_root+class_floder):
                for perspect_floder in os.listdir(data_root+class_floder+'/'+file_floder):
                    self.pc_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/pc.npy')
                    self.view_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/view.jpg')
                    self.pc_part_list.append(data_root + class_floder + '/' + file_floder + '/' + perspect_floder + '/part_far.npy')

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.view_list)}')


    def __getitem__(self, idx):
        views = self.transform(Image.open(self.view_list[idx]))
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        pc_part = np.load(self.pc_part_list[idx])[:1024].astype(np.float32)


        while pc.shape[0]<1040:
            pc = np.repeat(pc,2,axis=0)
            pc = pc[:self.pc_input_num]
        pc = pc/100
        pc = normal_pc(pc)
        # pc = np.transpose(pc)

        while pc_part.shape[0]<1024:
            pc_part = np.repeat(pc_part,2,axis=0)
            pc_part = pc_part[:1024]
        pc_part = pc_part/100
        pc_part = normal_pc(pc_part)
        # pc_part = np.transpose(pc_part)

        # pc = pc.transpose()
        # pc = np.expand_dims(pc.transpose(), axis=2)

        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()

    def __len__(self):
        return len(self.pc_list)

class GeneratorFetcher3392(Dataset):
    def __init__(self,data_root,status,pc_input_num=3392,data_sample=False):
        super(GeneratorFetcher3392,self).__init__()
        self.data_root = data_root
        self.view_list = []
        self.pc_list = []
        self.pc_part_list = []
        self.pc_input_num = pc_input_num
        self.status = status

        for class_floder in os.listdir(data_root):
            # print(class_floder)
            if data_sample == True:
                if class_floder not in ['12'] :
                    continue

            for file_floder in os.listdir(data_root+class_floder):
                for perspect_floder in os.listdir(data_root+class_floder+'/'+file_floder):
                    self.pc_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/pc.npy')
                    self.view_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/view.jpg')
                    self.pc_part_list.append(data_root + class_floder + '/' + file_floder + '/' + perspect_floder + '/part_far.npy')

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.view_list)}')


    def __getitem__(self, idx):
        views = self.transform(Image.open(self.view_list[idx]))
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        pc_part = np.load(self.pc_part_list[idx])[:1024].astype(np.float32)


        while pc.shape[0]<3392:
            pc = np.repeat(pc,2,axis=0)
            pc = pc[:self.pc_input_num]
        pc = pc/100
        pc = normal_pc(pc)
        # pc = np.transpose(pc)

        while pc_part.shape[0]<1024:
            pc_part = np.repeat(pc_part,2,axis=0)
            pc_part = pc_part[:1024]
        pc_part = pc_part/100
        pc_part = normal_pc(pc_part)
        # pc_part = np.transpose(pc_part)

        # pc = pc.transpose()
        # pc = np.expand_dims(pc.transpose(), axis=2)

        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()

    def __len__(self):
        return len(self.pc_list)

class GeneratorFetcherPSG3392(Dataset):
    def __init__(self,data_root,status,pc_input_num=3392,data_sample=False):
        super(GeneratorFetcherPSG3392,self).__init__()
        self.data_root = data_root
        self.view_list = []
        self.pc_list = []
        self.pc_input_num = pc_input_num
        self.status = status

        for class_floder in os.listdir(data_root):
            if data_sample == True:
                if class_floder not in ['12']:
                    continue

            for file_floder in os.listdir(data_root+class_floder):
                for perspect_floder in os.listdir(data_root+class_floder+'/'+file_floder):
                    self.pc_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/pc.npy')
                    self.view_list.append(data_root+class_floder+'/'+file_floder+'/'+perspect_floder+'/view.jpg')

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.view_list)}')


    def __getitem__(self, idx):
        views = self.transform(Image.open(self.view_list[idx]))
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)

        while pc.shape[0]<3392:
            pc = np.repeat(pc,2,axis=0)
            pc = pc[:self.pc_input_num]
        np.random.shuffle(pc)
        pc = pc[:self.pc_input_num]
        pc = normal_pc(pc)

        return views.float(), torch.from_numpy(pc).float()

    def __len__(self):
        return len(self.pc_list)

if __name__ == '__main__':
    # pkl = pickle.load(open('../02691156_1a32f10b20170883663e90eaf6b4ca52_05.dat', 'rb'))
    # img = pkl[0].astype('float32') / 255.0
    # label = pkl[1]

    psg = PSGPointViewFetcher('../data_test/',data_sample=True)

    # DataFetcher('../../../data/ModelNet40/pc','../../../data/ModelNet40/12_ModelNet40')