import json
from pprint import pprint
import os
import numpy as np
import trimesh
import math
import random
import time
import cv2

class Sampler():
    def __init__(self,to_do_list,thread_num):
        self.to_do_list = to_do_list
        self.dic = {}
        self.thread_num = thread_num

    def json_txt(self,dic_json):
        for items in dic_json:
            self.dic[items['id']] = items['objs']
            for key in items:
                if key == 'children':
                    self.json_txt(items['children'])

    def img_compare(self, do):
        [width, height, _] = do.shape
        for y in range(0, width):
            for x in range(0, height):
                if do[y, x][2] < 240 and do[y, x][2] > 0 and do[y, x][1] < 150 and do[y, x][1] > 0 and do[y, x][
                    0] < 150 and do[y, x][0] > 0:
                    pass
                else:
                    do[y, x] = [255, 255, 255]
        return do

    def main(self):
        to_do_list = self.to_do_list
        train_test_val_path = '../data/partnetdata/chair_geo'
        # save_path = '../data/partnetdata/chair_img_3000/'
        # to_do_list = []
        path = '/repository/zhangxc/data_v0'
        save_path = '/repository/zhangxc/chair_img_3000/'

        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # else:
        #     os.rmdir(save_path)
        #     os.mkdir(save_path)
        star_time = time.clock()
        count = 0
        for folder in to_do_list:
            count +=1
            json_name = os.path.join(path, folder, 'result_after_merging.json')
            with open(json_name, 'r') as f:
                data = json.load(f)
            self.json_txt(data)

            new_img = np.zeros(shape=(1, 400, 400, 3))
            for i in range(0, len(self.dic)):
                do = cv2.imread(os.path.join(path, folder, 'parts_render_after_merging', f'{i}.png'))
                img = self.img_compare(do)
                new_img = np.vstack((new_img, np.array(img)[np.newaxis, :]))
            new_img = np.delete(new_img, 0, 0)
            print(folder, 'finished')

            np.savez(os.path.join(save_path, f'{folder}.npz'), image=new_img)
            self.dic = {}


            print('Threading:', self.thread_num, 'Process:', count, '/', len(to_do_list), 'Time Used:',
                  time.clock() - star_time, 'Time Left:', (time.clock() - star_time) / count * (len(to_do_list) - count))



star_time = 0
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def run(file_list,thread_num):
    print(thread_num,'Start')
    count = 0

    sm = Sampler(file_list,thread_num)
    sm.main()
        # print(key,'Finished')
    # count += 1
    # print('Threading:',thread_num,'Process:',count,'/',len(file_list),'Time Used:',time.clock()-star_time,'Time Left:',(time.clock()-star_time)/count*(len(file_list)-count))


if __name__ == '__main__':

    save_path = '/repository/zhangxc/chair_img_3000/'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # else:
    #     os.rmdir(save_path)
    #     os.mkdir(save_path)

    thread_num = 60
    file_list = []
    exist_list = []
    for exist_file in os.listdir('/repository/zhangxc/chair_img_3000'):
        exist_list.append(exist_file)

    train_test_val_path = '../data/partnetdata/chair_geo'
    for file in os.listdir('../data/partnetdata/chair_geo'):
        if file[-3:]=='npz' and not exist_list.__contains__(file):
            file_list.append(file[:-4])
    print(len(file_list))
    print(file_list)


    file_list = chunks(file_list, thread_num)
    #
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(run, item, file_list.index(item)) for item in file_list]