import numpy as np

def rotate_point_cloud_by_angle_up_direction(pc, rotation_angle):
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

    rotated_pc = np.dot(pc.reshape((-1, 3)), rotation_matrix)
    rotated_pc=rotated_pc-np.mean(rotated_pc)

    return rotated_pc

def rotate_point_cloud_by_angle_straight_direction(pc, rotation_angle):
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

    rotated_pc = np.dot(pc.reshape((-1, 3)), rotation_matrix)
    rotated_pc = rotated_pc - np.mean(rotated_pc)

    return rotated_pc


if __name__=='__main__':
    data=np.load('173.npz')
    for key in data:
        print(key)

    pc=data['parts']

    for j in range(3):
        for i in range(8):
            res1 = rotate_point_cloud_by_angle_up_direction(pc, (7-i)*np.pi / 4)
            res=rotate_point_cloud_by_angle_straight_direction(res1,(1-j)*np.pi/6)

            pc1 = np.hstack((np.full([res.shape[0], 1], 'v'), res))
            np.savetxt(f'res{8*j+i+2}.obj', pc1, fmt='%s', delimiter=' ')



