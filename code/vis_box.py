import os
import matplotlib
from data import PartNetDataset
from vis_utils import draw_partnet_objects


if __name__ =='__main__':

    matplotlib.pyplot.ion()

    # visualize one data
    obj = PartNetDataset.load_object('/home/zhangxc/tmp/structurenet-master/data/results/pc_ae_chair_image_encoder_test/object-result.json')

    # edge visu: ADJ (red), ROT_SYM (yellow), TRANS_SYM (purple), REF_SYM (black)
    draw_partnet_objects(objects=[obj], object_names=['fig.json'],
                         figsize=(9, 5), leafs_only=True, visu_edges=True,
                         sem_colors_filename='../stats/semantics_colors/Chair.txt')

    print('PartNet Hierarchy: (the number in bracket corresponds to PartNet part_id)')
    print(obj)