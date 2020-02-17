# import os
# import matplotlib
# from data import PartNetDataset
# from vis_utils import draw_partnet_objects
#
# matplotlib.pyplot.ion()
#
# # ground-truth data directory
# root_dir = '../data/partnetdata/chair_hier'
#
# # read all data
# obj_list = sorted([int(item.split('.')[0]) for item in os.listdir(root_dir) if item.endswith('.json')])
#
# # visualize one data
# obj_id = 0
# obj = PartNetDataset.load_object(os.path.join(root_dir, str(obj_list[obj_id]) + '.json'), load_geo=True)
#
# # edge visu: ADJ (red), ROT_SYM (yellow), TRANS_SYM (purple), REF_SYM (black)
# draw_partnet_objects(objects=[obj], object_names=[str(obj_list[obj_id])],
#                      figsize=(9, 5), leafs_only=True, visu_edges=True, rep='geos',
#                      sem_colors_filename='../stats/semantics_colors/Chair.txt')
#
# print('PartNet Hierarchy: (the number in bracket corresponds to PartNet part_id)')
# print(obj)

import os
import matplotlib
from data import PartNetDataset
from vis_utils import draw_partnet_objects

matplotlib.pyplot.ion()

# results directory
root_dir = '/home/zhangxc/tmp/structurenet-master/data/results/pc_ae_chair_image_encoder_test'

# read all data
obj_list = sorted([int(item) for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item))])
# visualize one data
obj_id = 0
obj_dir = os.path.join(root_dir, str(obj_list[obj_id]))
print(obj_dir)

orig_obj = PartNetDataset.load_object(os.path.join(obj_dir, 'orig.json'))
recon_obj = PartNetDataset.load_object(os.path.join(obj_dir, 'recon.json'))

draw_partnet_objects(objects=[orig_obj, recon_obj], object_names=['original', 'reconstruction'],
                     figsize=(9, 5), leafs_only=True, visu_edges=True, rep='geos',
                     sem_colors_filename='../stats/semantics_colors/Chair.txt')

print('Original Structure:')
print(orig_obj)
print('Reconstructed Structure:')
print(recon_obj)

# import os
# import matplotlib
# from data import PartNetDataset
# from vis_utils import draw_partnet_objects
#
# matplotlib.pyplot.ion()
#
# # results directory
# root_dir = '../data/results/pc_vae_chair'
#
# # read all data
# obj_list = sorted([item for item in os.listdir(root_dir) if item.endswith('.json')])
#
# # visualize one data
# obj_id = 0
# obj = PartNetDataset.load_object(os.path.join(root_dir, obj_list[obj_id]))
#
# draw_partnet_objects(objects=[obj], object_names=[obj_list[obj_id]],
#                      figsize=(9, 5), leafs_only=True, visu_edges=True, rep='geos',
#                      sem_colors_filename='../stats/semantics_colors/Chair.txt')
#
# print('Tree Structure:')
# print(obj)