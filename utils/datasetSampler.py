import json
from pprint import pprint
import os
import numpy as np
import trimesh
import math
import random
import time


def triangle_area(v1, v2, v3):
    a = np.array(v2) - np.array(v1)
    b = np.array(v3) - np.array(v1)
    return np.sqrt(np.dot(a, a) * np.dot(b, b) - (np.dot(a, b) ** 2)) / 2.0


def cal_suface_area(mesh):
    areas = []
    for face in mesh.faces:
        v1, v2, v3 = face
        v1 = mesh.vertices[v1]
        v2 = mesh.vertices[v2]
        v3 = mesh.vertices[v3]

        areas += [triangle_area(v1, v2, v3)]
    areas = np.array(areas)
    return np.array(areas)


# sample_num = args.number
# output_file = args.output

def sampler(mesh, sample_num, normalize):
    # output = open(output_file, 'w')

    # mesh = trimesh.load(input_file)
    # print("number of vertices: ", len(mesh.vertices))
    # print("number of faces: ", len(mesh.faces))

    areas = cal_suface_area(mesh)
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


def json_txt(dic_json):
    for items in dic_json:
        dic[items['id']] = items['objs']
        for key in items:
            if key == 'children':
                json_txt(items['children'])


if __name__ == "__main__":
    train_test_val_path = '../data/partnetdata/chair_geo'
    save_path = '../data/partnetdata/chair_geo_3000/'
    to_do_list = []
    path = '/repository/zhangxc/data_v0'
    start = time.time()

    with open(os.path.join(train_test_val_path,'train_no_other_less_than_10_parts.txt'),'r') as f:
        line = f.readline()
        line = line[:-1]
        while line:
            line = f.readline()
            line = line[:-1]
            to_do_list.append(line)

    with open(os.path.join(train_test_val_path,'test_no_other_less_than_10_parts.txt'),'r') as f:
        line = f.readline()
        line = line[:-1]
        while line:
            line = f.readline()
            line = line[:-1]
            to_do_list.append(line)

    with open(os.path.join(train_test_val_path,'val_no_other_less_than_10_parts.txt'),'r') as f:
        line = f.readline()
        line = line[:-1]
        while line:
            line = f.readline()
            line = line[:-1]
            to_do_list.append(line)

    print(len(to_do_list))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        os.rmdir(save_path)
        os.mkdir(save_path)

    count = 0
    for folder in os.listdir(path):
        if not to_do_list.__contains__(folder):
            continue
        count+=1
        dic = {}
        json_name = os.path.join(path, folder, 'result_after_merging.json')
        with open(json_name, 'r') as f:
            data = json.load(f)
        json_txt(data)
        mesh_dict_src = {}
        for mesh_name in os.listdir(os.path.join(path, folder, 'objs')):
            mesh_path = os.path.join(path, folder, 'objs', mesh_name)
            mesh_dict_src[mesh_name.split('.')[0]] = trimesh.load_mesh(mesh_path)

        new_point = np.zeros(shape=(1, 3000, 3))
        # print(new_point.shape)
        for key in dic:
            new_mesh = trimesh.Trimesh(vertices=[], faces=[])
            for mesh_name in dic[key]:
                new_mesh = new_mesh + mesh_dict_src[mesh_name]
            points = sampler(new_mesh, 3000, False)
            # print(np.array(points)[np.newaxis,:].shape)
            new_point = np.vstack((new_point, np.array(points)[np.newaxis, :]))
        new_point = np.delete(new_point, 0, 0)
        print(folder, 'finished')
        np.savez(os.path.join(save_path, f'{folder}.npz'), parts=new_point)
        # print(time.strftime('%H%M%S',time.gmtime(time.time()-start)))



