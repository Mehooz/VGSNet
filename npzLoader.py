import numpy as np
import os
import json


if __name__ == "__main__":
    root_path='/home/zhangxc/tmp/structurenet-master/data/results/pc_ae_chair'
    list=os.listdir('/home/zhangxc/tmp/structurenet-master/data/results/pc_ae_chair')
    count=0
    for idx in list:
        count+=1
        if count>10:
            break
        with open(root_path+'/'+idx+'/recon.json') as f:
            js_data=json.load(f)

        data = np.array(js_data['geo'])
        data=data.reshape(1000,3)
        pc = np.hstack((np.full([data.shape[0], 1], 'v'), data))
        np.savetxt(root_path+f'/{idx}.obj', pc, fmt='%s', delimiter=' ')