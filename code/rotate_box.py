import numpy as np
import json
import copy
import os


def rotate_box_by_angle_up_direction(box, rotation_angle):

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    box[0:3] = np.dot(box[0:3], rotation_matrix)
    box[6:9] = np.dot(box[6:9], rotation_matrix)
    box[9:] = np.dot(box[9:], rotation_matrix)

    return box


def rotate_box_by_angle_straight_direction(box, rotation_angle):

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0,cosval, sinval],
                                [0,-sinval, cosval]])



    box[0:3] = np.dot(box[0:3], rotation_matrix)
    box[6:9] = np.dot(box[6:9], rotation_matrix)
    box[9:] = np.dot(box[9:], rotation_matrix)



    return box


def rotate_edge_by_angle_up_direction(edge, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array
        Return:
          Nx3 array
    """
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    edge[0:3] = np.dot(edge[0:3], rotation_matrix)
    edge[3:6] = np.dot(edge[3:6], rotation_matrix)


    return edge


def rotate_edge_by_angle_straight_direction(edge, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array
        Return:
          Nx3 array
    """
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0,cosval, sinval],
                                [0,-sinval, cosval]])

    edge[0:3] = np.dot(edge[0:3], rotation_matrix)
    edge[3:6] = np.dot(edge[3:6], rotation_matrix)


    return edge


def recursive_read(x, up_ang, straight_ang,center_x,center_y):

    x['box'] =rotate_box_by_angle_up_direction(x['box'],up_ang)
    x['box'] =rotate_box_by_angle_straight_direction(x['box'],straight_ang)

    if 'edges' in x:
        for i in range(len(x['edges'])):

            if 'params' in x['edges'][i] and len(x['edges'][i]['params']) >= 6:
                x['edges'][i]['params'] = rotate_edge_by_angle_up_direction(x['edges'][i]['params'], up_ang)
                x['edges'][i]['params'] = rotate_edge_by_angle_straight_direction(x['edges'][i]['params'], straight_ang)


    if not 'children' in x:
        return

    for i in range(len(x['children'])):
        recursive_read(x['children'][i],up_ang, straight_ang,center_x,center_y)


if __name__=='__main__':
    ROOT_PATH='../data/partnetdata/storagefurniture_hier_3000/'
    TARGET_PATH='../data/partnetdata/storagefurniture_hier_new/'
    list=os.listdir(ROOT_PATH)

    count=0
    print(len(list))

    for f_name in list:
        count+=1
        if count%200==0:
            print(count)
        if f_name[-5:]!='.json':
            continue
        name=ROOT_PATH+f_name
        with open(name) as f:
            ori_data = json.load(f)
        TARGET_NAME=TARGET_PATH+f_name
        os.mkdir(TARGET_NAME[:-5])

        center_x = ori_data['box'][6:9]
        center_y = ori_data['box'][9:]

        for j in range(3):
            for i in range(8):
                data = copy.deepcopy(ori_data)
                recursive_read(data, (7 - i) * np.pi / 4, (1- j) * np.pi / 6, center_x, center_y)
                with open(TARGET_NAME[:-5]+'/'+f_name[:-5]+f'_{8*j+i+1}.json', 'w') as f:
                    json.dump(data, f)

