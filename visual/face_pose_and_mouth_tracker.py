import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import math

# --- ROS2 和 转换库 ---
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

class FacePosePublisher(Node):
    def __init__(self):
        super().__init__('face_pose_publisher')
        self.publisher_ = self.create_publisher(Pose, '/face_pose', 10)
        
        # --- 1. 初始化模块 ---
        self.init_realsense()
        self.init_mediapipe()

        # --- 2. 定义3D人脸模型 (仅用于solvePnP姿态估计) ---
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # 鼻尖 (Nose tip) - 1
            (0.0, -3.30, -6.30),         # 下巴 (Chin) - 152
            (-2.25, 1.70, -4.80),        # 左眼左角 (Left eye left corner) - 33
            (2.25, 1.70, -4.80),         # 右眼右角 (Right eye right corner) - 263
            (-1.50, -1.50, -5.20),       # 左嘴角 (Left Mouth corner) - 61
            (1.50, -1.50, -5.20)         # 右嘴角 (Right mouth corner) - 291
        ], dtype=np.float64)
        self.model_points_indices = [1, 152, 33, 263, 61, 291]
        
        # 目标点使用嘴部中心
        self.mouth_indices = [13, 14, 78, 308] 
        
        self.get_logger().info("人脸位姿发布节点已启动...")

    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[self.intr.fx, 0, self.intr.ppx],
                                       [0, self.intr.fy, self.intr.ppy],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(4)

    def init_mediapipe(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def run_loop(self):
        """主循环，采用混合方法处理图像并发布"""
        while rclpy.ok():
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            img_h, img_w, _ = color_image.shape

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=color_image, landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

                    # --- 步骤 1: 使用 solvePnP 获取姿态 (Rotation) ---
                    image_points_2d = np.array([
                        (int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h))
                        for idx in self.model_points_indices
                    ], dtype=np.float64)

                    # 我们只关心 rotation_vector，tvec反投影的位置仅用于可视化
                    (success, rotation_vector, tvec_for_vis) = cv2.solvePnP(
                        self.model_points_3d, image_points_2d, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if not success:
                        continue
                        
                    # --- 步骤 2: 使用深度相机获取精确的位置 (Position) ---
                    mouth_points_2d = [ (int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h)) for idx in self.mouth_indices ]
                    mouth_center_2d = np.mean(mouth_points_2d, axis=0).astype(int)
                    
                    # 钳位坐标以防止越界错误
                    mouth_center_2d[0] = np.clip(mouth_center_2d[0], 0, img_w - 1)
                    mouth_center_2d[1] = np.clip(mouth_center_2d[1], 0, img_h - 1)
                    
                    depth = depth_frame.get_distance(mouth_center_2d[0], mouth_center_2d[1])

                    # 只有在深度有效时才继续
                    if depth > 0:
                        # 通过反投影得到相机坐标系下的精确3D位置 (单位: 米)
                        pos_in_camera_frame = rs.rs2_deproject_pixel_to_point(
                            self.intr, [mouth_center_2d[0], mouth_center_2d[1]], depth
                        )
                        
                        # --- 步骤 3: 组合精确位置和姿态，并进行坐标变换 ---
                        pos_x, pos_y, pos_z, quat = self.transform_hybrid_pose_to_robot_frame(
                            rotation_vector, pos_in_camera_frame
                        )
                        
                        # --- 步骤 4: 发布最终的位姿指令 ---
                        pose_msg = Pose()
                        pose_msg.position.x = pos_x
                        pose_msg.position.y = pos_y
                        pose_msg.position.z = pos_z
                        # 保持正确的 [x, y, z, w] 顺序
                        pose_msg.orientation.x = quat[0]
                        pose_msg.orientation.y = quat[1]
                        pose_msg.orientation.z = quat[2]
                        pose_msg.orientation.w = quat[3]
                        
                        self.publisher_.publish(pose_msg)
                        self.get_logger().info(f'发布位姿: Pos(x={pos_x:.2f}, y={pos_y:.2f}, z={pos_z:.2f}) Quat(x={quat[0]:.2f}, y={quat[1]:.2f}, z={quat[2]:.2f}, w={quat[3]:.2f})')

                    # --- 可视化部分 (仍然使用旧数据来保证显示正确性) ---
                    mouth_3d_coords_cm = np.array(pos_in_camera_frame) * 100 if depth > 0 else None
                    self.visualize_all_info(color_image, rotation_vector, tvec_for_vis,
                                            mouth_center_2d, mouth_3d_coords_cm)

            cv2.imshow('Face Pose Publisher', color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            rclpy.spin_once(self, timeout_sec=0.001)

    def transform_hybrid_pose_to_robot_frame(self, rvec, pos_cam):
        """
        将混合位姿(solvePnP的姿态 + 深度相机的位置)转换为ROS机器人坐标系。
        :param rvec:来自 solvePnP 的旋转向量。
        :param pos_cam:来自深度相机反投影的3D位置向量 [x, y, z] (单位: 米)。
        """
        # --- 位置变换 ---
        # 坐标系映射: Camera(X右,Y下,Z前) -> ROS(X前,Y左,Z上)
        pos_x =  pos_cam[2]
        pos_y = -pos_cam[0]
        pos_z = -pos_cam[1]
        
        # --- 姿态变换 (与之前相同) ---
        r = R.from_rotvec(rvec.flatten())
        cam_to_robot_rotation = R.from_euler('zx', [-90, -90], degrees=True)
        final_rotation = r * cam_to_robot_rotation
        quat = final_rotation.as_quat() # [x, y, z, w]
        
        return pos_x, pos_y, pos_z, quat

    def visualize_all_info(self, image, rvec, tvec, mouth_center_2d, mouth_3d_coords_cm):
        """在图像上绘制所有需要的信息"""
        # 绘制位姿坐标轴 (使用 solvePnP 的 tvec 来确保轴画在人脸模型中心)
        axis_points_3d = np.array([(0,0,0), (5,0,0), (0,5,0), (0,0,5)], dtype=np.float64)
        axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axis_points_2d = axis_points_2d.astype(int).reshape(-1, 2)
        cv2.line(image, axis_points_2d[0], axis_points_2d[1], (255,0,0), 3)
        cv2.line(image, axis_points_2d[0], axis_points_2d[2], (0,255,0), 3)
        cv2.line(image, axis_points_2d[0], axis_points_2d[3], (0,0,255), 3)

        # 绘制精确的嘴部中心点
        cv2.circle(image, tuple(mouth_center_2d), 5, (0, 0, 255), -1)

        # 显示文本信息
        mouth_text = f"Mouth Pos (cm): {np.array2string(mouth_3d_coords_cm, precision=1)}" \
                     if mouth_3d_coords_cm is not None else "Mouth Pos (cm): N/A"
        cv2.putText(image, mouth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        cv2.putText(image, f"Pitch: {angles[0]:.1f}, Yaw: {angles[1]:.1f}, Roll: {angles[2]:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def cleanup(self):
        self.pipeline.stop()
        self.face_mesh.close()
        cv2.destroyAllWindows()
        self.get_logger().info("资源已清理，节点关闭。")

def main(args=None):
    rclpy.init(args=args)
    face_pose_publisher = FacePosePublisher()
    try:
        face_pose_publisher.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        face_pose_publisher.cleanup()
        face_pose_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()