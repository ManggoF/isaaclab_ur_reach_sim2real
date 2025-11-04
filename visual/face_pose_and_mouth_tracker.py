import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import math

# --- 1. 初始化所有模块 ---

# 初始化RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取深度传感器的测量单位（米）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 创建对齐对象，将深度图与彩色图对齐
align_to = rs.stream.color
align = rs.align(align_to)

# 获取相机内参
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(4) # 假设没有畸变

# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# --- 2. 定义3D人脸模型和关键点索引 ---

# 这是solvePnP的关键：一个标准化的3D人脸模型点。
# 这些点是基于MediaPipe文档中的通用人脸模型选择的。单位是厘米(cm)。
# 我们不需要完整的478个点，只需要几个分布均匀、稳定的点即可。
model_points_3d = np.array([
    (0.0, 0.0, 0.0),             # 鼻尖 (Nose tip) - 1
    (0.0, -3.30, -6.30),         # 下巴 (Chin) - 152
    (-2.25, 1.70, -4.80),        # 左眼左角 (Left eye left corner) - 33
    (2.25, 1.70, -4.80),         # 右眼右角 (Right eye right corner) - 263
    (-1.50, -1.50, -5.20),       # 左嘴角 (Left Mouth corner) - 61
    (1.50, -1.50, -5.20)         # 右嘴角 (Right mouth corner) - 291
], dtype=np.float64)

# 对应于上面3D点的MediaPipe关键点索引
model_points_indices = [1, 152, 33, 263, 61, 291]

# 嘴部中心计算所需的关键点索引 (上下左右唇的中心点)
mouth_indices = [13, 14, 78, 308] 


print("Starting stream... Press 'q' to quit.")

try:
    while True:
        # --- 3. 获取并处理图像 ---
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        
        # 为了提高性能，将图像标记为不可写
        color_image.flags.writeable = False
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # 运行MediaPipe Face Mesh
        results = face_mesh.process(image_rgb)
        
        # 恢复图像可写
        color_image.flags.writeable = True
        
        img_h, img_w, _ = color_image.shape
        face_landmarks_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # --- 4. 提取2D关键点用于位姿估计 ---
                image_points_2d = []
                for idx in model_points_indices:
                    loc = face_landmarks.landmark[idx]
                    x, y = int(loc.x * img_w), int(loc.y * img_h)
                    image_points_2d.append((x, y))
                
                image_points_2d = np.array(image_points_2d, dtype=np.float64)

                # --- 5. 使用solvePnP计算位姿 ---
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    model_points_3d, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # --- 6. 计算嘴部中心的3D坐标 ---
                    mouth_points_2d = []
                    for idx in mouth_indices:
                        loc = face_landmarks.landmark[idx]
                        x, y = int(loc.x * img_w), int(loc.y * img_h)
                        mouth_points_2d.append((x,y))
                    
                    mouth_center_2d = np.mean(mouth_points_2d, axis=0).astype(int)
                    
                    # 从深度图中获取深度值
                    mouth_depth = depth_frame.get_distance(mouth_center_2d[0], mouth_center_2d[1])

                    mouth_3d_coords = None
                    if mouth_depth > 0: # 确保深度值有效
                        # 使用RealSense SDK的反投影函数
                        mouth_3d_coords = rs.rs2_deproject_pixel_to_point(
                            intr, [mouth_center_2d[0], mouth_center_2d[1]], mouth_depth
                        )
                        # 结果单位是米
                        mouth_3d_coords = np.array(mouth_3d_coords) * 100 # 转换为厘米(cm)方便显示

                    # --- 7. 可视化结果 ---
                    # 绘制人脸网格
                    mp_drawing.draw_landmarks(
                        image=color_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

                    # 绘制坐标轴以显示位姿
                    axis_points_3d = np.array([(0,0,0), (5,0,0), (0,5,0), (0,0,5)], dtype=np.float64)
                    axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    axis_points_2d = axis_points_2d.astype(int).reshape(-1, 2)
                    
                    # 绘制位姿坐标轴
                    cv2.line(color_image, axis_points_2d[0], axis_points_2d[1], (255,0,0), 3) # X轴-红色
                    cv2.line(color_image, axis_points_2d[0], axis_points_2d[2], (0,255,0), 3) # Y轴-绿色
                    cv2.line(color_image, axis_points_2d[0], axis_points_2d[3], (0,0,255), 3) # Z轴-蓝色

                    # 绘制嘴部中心点
                    cv2.circle(color_image, tuple(mouth_center_2d), 5, (0,0,255), -1)

                    # 显示信息
                    cv2.putText(color_image, f"Mouth Pos (cm): {mouth_3d_coords}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # 将旋转向量转换为欧拉角 (Roll, Pitch, Yaw)
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    cv2.putText(color_image, f"Pitch: {angles[0]:.1f}, Yaw: {angles[1]:.1f}, Roll: {angles[2]:.1f}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


        cv2.imshow('RealSense Face Pose', color_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    # --- 8. 清理 ---
    pipeline.stop()
    face_mesh.close()
    cv2.destroyAllWindows()