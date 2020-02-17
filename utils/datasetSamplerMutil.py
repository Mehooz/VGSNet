import json
from pprint import pprint
import os
import numpy as np
import trimesh
import math
import random
import time

class Sampler():
    def __init__(self,to_do_list,thread_num):
        self.to_do_list = to_do_list
        self.dic = {}
        self.thread_num = thread_num

    def triangle_area(self,v1, v2, v3):
        a = np.array(v2) - np.array(v1)
        b = np.array(v3) - np.array(v1)
        return np.sqrt(np.abs(np.dot(a, a) * np.dot(b, b) - (np.dot(a, b) ** 2))) / 2.0

    def cal_suface_area(self,mesh):
        areas = []
        for face in mesh.faces:
            v1, v2, v3 = face
            v1 = mesh.vertices[v1]
            v2 = mesh.vertices[v2]
            v3 = mesh.vertices[v3]

            areas += [self.triangle_area(v1, v2, v3)]
        areas = np.array(areas)
        return np.array(areas)

    # sample_num = args.number
    # output_file = args.output

    def sampler(self,mesh, sample_num, normalize):
        # output = open(output_file, 'w')

        # mesh = trimesh.load(input_file)
        # print("number of vertices: ", len(mesh.vertices))
        # print("number of faces: ", len(mesh.faces))

        areas = self.cal_suface_area(mesh)
        prefix_sum = np.cumsum(areas)

        total_area = prefix_sum[-1]
        # print("total area: ", total_area)

        sample_points = []
        for i in range(sample_num):
            prob = random.random()
            sample_pos = prob * total_area

            # Here comes the binary search
            left_bound, right_bound = 0, len(areas) - 1
            while left_bound < right_bound:
                mid = (left_bound + right_bound) // 2
                if sample_pos <= prefix_sum[mid]:
                    right_bound = mid
                else:
                    left_bound = mid + 1

            target_surface = right_bound

            # Sample point on the surface
            v1, v2, v3 = mesh.faces[target_surface]

            v1 = mesh.vertices[v1]
            v2 = mesh.vertices[v2]
            v3 = mesh.vertices[v3]

            edge_vec1 = np.array(v2) - np.array(v1)
            edge_vec2 = np.array(v3) - np.array(v1)

            prob_vec1, prob_vec2 = random.random(), random.random()
            if prob_vec1 + prob_vec2 > 1:
                prob_vec1 = 1 - prob_vec1
                prob_vec2 = 1 - prob_vec2

            target_point = np.array(v1) + (edge_vec1 * prob_vec1 + edge_vec2 * prob_vec2)
            # Random picking point in a triangle: http://mathworld.wolfram.com/TrianglePointPicking.html

            sample_points.append(target_point)

        if normalize:
            print('Apply normalization to unit ball')
            norms = np.linalg.norm(sample_points, axis=1)
            max_norm = max(norms)
            print('max norm: ', max_norm)
            sample_points /= max_norm

        return sample_points
        # for points in sample_points:
        #     output.write( ' '.join(["%.4f" % _ for _ in points]) )
        #     output.write('\n')

    def json_txt(self,dic_json):
        for items in dic_json:
            self.dic[items['id']] = items['objs']
            for key in items:
                if key == 'children':
                    self.json_txt(items['children'])

    def main(self):
        to_do_list = self.to_do_list
        exist_list = os.listdir('/repository/zhangxc/chair_img_3000')
        train_test_val_path = '../data/partnetdata/chair_geo'
        save_path = '../data/partnetdata/chair_geo_3000/'
        # to_do_list = []
        path = '/repository/zhangxc/data_v0'

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
            mesh_dict_src = {}
            for mesh_name in os.listdir(os.path.join(path, folder, 'objs')):
                mesh_path = os.path.join(path, folder, 'objs', mesh_name)
                mesh_dict_src[mesh_name.split('.')[0]] = trimesh.load_mesh(mesh_path)

            new_point = np.zeros(shape=(1, 3000, 3))
            # print(new_point.shape)
            for key in self.dic:
                new_mesh = trimesh.Trimesh(vertices=[], faces=[])
                for mesh_name in self.dic[key]:
                    new_mesh = new_mesh + mesh_dict_src[mesh_name]
                points = self.sampler(new_mesh, 3000, False)
                # print(np.array(points)[np.newaxis,:].shape)
                new_point = np.vstack((new_point, np.array(points)[np.newaxis, :]))
            new_point = np.delete(new_point, 0, 0)
            # print(folder, 'finished')

            np.savez(os.path.join(save_path, f'{folder}.npz'), parts=new_point)
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
    save_path = '../data/partnetdata/chair_geo_3000/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        os.rmdir(save_path)
        os.mkdir(save_path)

    thread_num = 100
    file_list = []
    train_test_val_path = '../data/partnetdata/chair_geo'
    for file in os.listdir('../data/partnetdata/chair_geo'):
        if file[-3:]=='npz':
            file_list.append(file[:-4])
    print(len(file_list))
    print(file_list)
    # with open(os.path.join(train_test_val_path,'train.txt'),'r') as f:
    #     line = f.readline()
    #     line = line[:-1]
    #     while line:
    #         line = f.readline()
    #         line = line[:-1]
    #         file_list.append(line)
    #
    # with open(os.path.join(train_test_val_path,'test.txt'),'r') as f:
    #     line = f.readline()
    #     line = line[:-1]
    #     while line:
    #         line = f.readline()
    #         line = line[:-1]
    #         file_list.append(line)
    #
    # with open(os.path.join(train_test_val_path,'val.txt'),'r') as f:
    #     line = f.readline()
    #     line = line[:-1]
    #     while line:
    #         line = f.readline()
    #         line = line[:-1]
    #         file_list.append(line)

    file_list = chunks(file_list, thread_num)
    #
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(run, item, file_list.index(item)) for item in file_list]