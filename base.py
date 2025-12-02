#!/usr/bin/env python

import numpy as np
import cv2

from scipy.spatial.transform import Rotation


def pose_to_matrix(position, orientation):
    """
    Convert a geometry_msgs/Pose to a 4x4 numpy homogeneous transformation matrix.
    Args: pose (Pose): ROS geometry_msgs/Pose message.
    Returns: np.ndarray: 4x4 homogeneous transformation matrix.
    """
    # # Extract position
    # position = np.array([pose.position.x, pose.position.y, pose.position.z])
    # # Extract orientation as quaternion
    # quaternion = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    # Convert quaternion to rotation matrix
    translation = [position.x, position.y, position.z]
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    # Construct the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    return transformation_matrix

def matrix_to_pose(matrix):
    """
    Convert a 4x4 homogeneous transformation matrix to a geometry_msgs/Pose.
    Args: matrix (np.ndarray): 4x4 homogeneous transformation matrix.
    Returns: Pose: ROS geometry_msgs/Pose message.
    """
    # Ensure the input is a 4x4 matrix
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix"
    # Extract position
    position = matrix[:3, 3]
    # Extract rotation matrix and convert to quaternion
    rotation_matrix = matrix[:3, :3]
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
    # # Create Pose message
    # pose = Pose()
    # pose.position.x = position[0]
    # pose.position.y = position[1]
    # pose.position.z = position[2]
    # pose.orientation.x = quaternion[0]
    # pose.orientation.y = quaternion[1]
    # pose.orientation.z = quaternion[2]
    # pose.orientation.w = quaternion[3]

    return position,quaternion

def dist_position(position1, position2):
    return ((position1.x - position2.x) * (position1.x - position2.x) +(position1.y - position2.y) * (position1.y - position2.y)) ** 0.5

# 两个四元数之间 转换为姿态矩阵后 X轴的角度差
def angle_quaternions_x(orientation1, orientation2):
    quaternion1 = [orientation1.x, orientation1.y, orientation1.z, orientation1.w]
    rotation_matrix1 = Rotation.from_quat(quaternion1).as_matrix()
    quaternion2 = [orientation2.x, orientation2.y, orientation2.z, orientation2.w]
    rotation_matrix2 = Rotation.from_quat(quaternion2).as_matrix()

    R_2_2_1 = np.linalg.inv(rotation_matrix1) @ rotation_matrix2

    direction_xaxis = np.array([R_2_2_1[0,0],R_2_2_1[1,0]])
    direction_xaxis = direction_xaxis / np.linalg.norm(direction_xaxis)

    if direction_xaxis[0] >= 0:
        angle_xaxis = np.arcsin(direction_xaxis[1])
    elif direction_xaxis[0] < 0 and direction_xaxis[1] >= 0:
        angle_xaxis = 3.14159 - np.arcsin(direction_xaxis[1])
    elif direction_xaxis[0] < 0 and direction_xaxis[1] < 0:
        angle_xaxis = -3.14159 - np.arcsin(direction_xaxis[1])

    # angle_xaxis = np.arcsin(direction_xaxis[1])
    
    return angle_xaxis

# 一个四元数的x轴 一个方向向量 之间的角度差
def angle_quaternionX_vector(orientation, vec):
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rotation_matrix1 = Rotation.from_quat(quaternion).as_matrix()
    
    direction_xaxis = np.array([rotation_matrix1[0,0],rotation_matrix1[1,0]])
    v1 = direction_xaxis / np.linalg.norm(direction_xaxis)

    direction_point = np.array([vec[0],vec[1]])
    v2 = direction_point / np.linalg.norm(direction_point)
    
    # 两个方向向量的夹角
    # 计算两个向量的点积
    dot_product = np.dot(v1, v2)
    
    # 计算两个向量的模长
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # 为了避免浮点数误差，使 cos_theta 的值在 -1 和 1 之间
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 计算夹角（弧度）
    angle = np.arccos(cos_theta)
    
    return angle


# ur获取的平移向量和旋转向量转换成位姿矩阵
def vect_to_matrix(x, y, z, rx, ry, rz):
    theta = np.linalg.norm([rx, ry, rz])
    if theta < 1e-6:  # If the angle is very small, treat as no rotation
        R = np.eye(3)
    else:
        # Normalize the rotation vector
        r = np.array([rx, ry, rz]) / theta        
        # Compute Rodrigues' rotation formula components
        K = np.array([[0, -r[2], r[1]],
                    [r[2], 0, -r[0]],
                    [-r[1], r[0], 0]])  # Skew-symmetric matrix of r
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        
    rotation_vector = np.array([rx, ry, rz])  # Rotation vector (axis-angle)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    t = np.array([x, y, z])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return R, t, T


# 位姿矩阵转换为ur平移向量和旋转向量
def matrix_to_vect(T):
    t = T[:3, 3]
    R = T[:3, :3]
    rotation_vector, _ = cv2.Rodrigues(R)

    return t[0], t[1], t[2], float(rotation_vector[0]), float(rotation_vector[1]), float(rotation_vector[2])


# 根据深度图中目标区域的左上右下坐标，以及有效区域比例，计算对应区域点云的质心和法向量
def cal_centroidNormalfromDepthRegion(depth_image, x_min, y_min, x_max, y_max, fx, fy, cx, cy, valid_ratio=1.0):
    """
    根据深度图中的目标区域的左上和右下坐标，以及有效区域比例，计算点云的质心和法向量。
    
    :param depth_image: 输入的深度图 (2D numpy array)
    :param top_left: 目标区域左上角坐标 (y_min, x_min)
    :param bottom_right: 目标区域右下角坐标 (y_max, x_max)
    :param valid_ratio: 有效区域比例，决定了需要考虑的区域比例 (默认值是1.0，表示使用整个区域)
    
    :return: (质心坐标, 法向量)
    """

   # 计算目标区域的中心点
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2
    
    # 计算目标区域的高度和宽度
    height = y_max - y_min
    width = x_max - x_min
    
    # 根据有效区域比例计算新的区域尺寸
    new_height = int(height * valid_ratio)
    new_width = int(width * valid_ratio)

    # 计算新的区域的上下左右边界，确保不超出图像边界
    new_y_min = max(center_y - new_height // 2, 0)
    new_y_max = min(center_y + new_height // 2, depth_image.shape[0])
    new_x_min = max(center_x - new_width // 2, 0)
    new_x_max = min(center_x + new_width // 2, depth_image.shape[1])

    # 提取新的有效区域的深度图数据
    region = depth_image[new_y_min:new_y_max, new_x_min:new_x_max]
    # 生成像素网格（u, v）
    u, v = np.meshgrid(np.arange(new_x_min, new_x_max), np.arange(new_y_min, new_y_max))

    # 将像素坐标 (u, v) 转换为 (X, Y, Z)
    Z = region.astype(float)  # 深度值，单位通常是毫米
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # 获取有效的像素（深度值不为0）
    valid_points = np.argwhere(Z > 0)
    
    # 计算质心
    valid_depth_values_z = Z[valid_points[:, 0], valid_points[:, 1]]
    valid_depth_values_x = X[valid_points[:, 0], valid_points[:, 1]]
    valid_depth_values_y = Y[valid_points[:, 0], valid_points[:, 1]]
    
    # 将二维坐标转换为点云（x, y, z）
    points = np.zeros((valid_points.shape[0], 3))
    points[:, 0] = valid_depth_values_x  # x
    points[:, 1] = valid_depth_values_y  # y
    points[:, 2] = valid_depth_values_z  # z (深度值)

    # 计算质心 (x, y, z)
    centroid = np.mean(points, axis=0)
    
    # 计算法向量 (计算三个邻域点的法向量)
    # 计算梯度（用于法向量计算）
    dx = np.gradient(region, axis=1)  # x方向的梯度
    dy = np.gradient(region, axis=0)  # y方向的梯度

    # 获取有效的梯度区域
    valid_gradients_x = dx[valid_points[:, 0], valid_points[:, 1]]
    valid_gradients_y = dy[valid_points[:, 0], valid_points[:, 1]]

    # 使用深度值计算法向量
    normals = np.zeros((valid_points.shape[0], 3))
    normals[:, 0] = valid_gradients_x  # x梯度
    normals[:, 1] = valid_gradients_y  # y梯度
    normals[:, 2] = -np.ones(valid_points.shape[0])  # z方向法向量（向下，深度为负）

    # 归一化法向量
    norm_lengths = np.linalg.norm(normals, axis=1)
    normals = normals / norm_lengths[:, np.newaxis]

    # 法向量是所有有效法向量的平均
    average_normal = np.mean(normals, axis=0)

    return centroid, average_normal


def cal_pointTransform(transformation_matrix, point):
    """
    根据位姿变换矩阵对坐标点进行变换。
    
    :param transformation_matrix: 4x4 位姿变换矩阵 (齐次变换矩阵)
    :param point: 3D 坐标点 (x, y, z)，表示为一个包含 3 个元素的数组或列表
    :return: 变换后的 3D 坐标点 (x', y', z')
    """
    # 将点表示为齐次坐标 (x, y, z, 1)
    point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
    
    # 使用变换矩阵进行矩阵乘法
    transformed_point = np.dot(transformation_matrix, point_homogeneous)
    
    # 返回变换后的坐标 (x', y', z')
    return transformed_point[:3]

def cal_normalTransform(transformation_matrix, normal):
    """
    根据旋转矩阵变换法向量。
    
    :param rotation_matrix: 3x3 旋转矩阵
    :param normal: 法向量 (x, y, z)，表示为一个包含 3 个元素的数组或列表
    :return: 变换后的法向量 (x', y', z')
    """
    rotation_matrix = transformation_matrix[:3, :3]
    
    # 将法向量转换为 numpy 数组
    normal = np.array(normal)
    
    # 使用旋转矩阵变换法向量
    transformed_normal = np.dot(rotation_matrix, normal)
    
    # 返回变换后的法向量
    return transformed_normal

