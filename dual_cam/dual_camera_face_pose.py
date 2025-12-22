import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import math
from std_msgs.msg import Float32

# --- ROS2 和 转换库 ---
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor # 引入多线程执行器
from geometry_msgs.msg import PoseStamped 
from scipy.spatial.transform import Rotation as R

class FacePosePublisher(Node):
    def __init__(self, node_name, camera_frame_id, publish_topic, serial_number):
        # 节点名称必须在初始化时传递
        super().__init__(node_name)
        
        # --- 0. 参数配置 (直接通过构造函数传递，不再使用declare_parameter) ---
        self.camera_frame_id = camera_frame_id
        self.publish_topic = publish_topic
        self.serial_number = serial_number
        
        # 使用配置的话题创建发布器
        self.publisher_ = self.create_publisher(PoseStamped, self.publish_topic, 10)
        # 【新增】嘴巴距离发布器
        # 话题名: /{node_name}/mouth_dist
        self.mouth_dist_publisher_ = self.create_publisher(Float32, f'/{node_name}/mouth_dist', 10)
        # 【新增】用于存储最新的图像给主线程显示
        self.latest_image = None
        
        # --- 1. 初始化模块 ---
        self.init_realsense()
        self.init_mediapipe()

        # --- 2. 定义3D人脸模型 (仅用于solvePnP姿态估计) ---
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # 鼻尖 (Nose tip) - 1
            (0.0, 85, -60),         # 下巴 (Chin) - 152
            (65, -45, -50),        # 右眼右角 (Left eye left corner) - 33
            (-65, -45, -50),         # 左眼左角 (Right eye right corner) - 263
            (25, 40, -40),       # 右嘴角 (Left Mouth corner) - 61
            (-25, 40, -40)         # 左嘴角 (Right mouth corner) - 291
        ], dtype=np.float64)
        self.model_points_indices = [1, 152, 33, 263, 61, 291]
        
        # 目标点使用嘴部中心
        self.mouth_indices = [13, 14, 78, 308] 
        # 【新增】用于计算嘴唇垂直距离的地标点索引
        self.TOP_LIP_IDX = 13
        self.BOTTOM_LIP_IDX = 14
        
        self.get_logger().info(f"人脸位姿发布节点 '{node_name}' 已启动. 序列号: {serial_number}, 帧: {self.camera_frame_id}, 话题: {self.publish_topic}...")

        # 创建一个定时器来代替 run_loop 中的无限循环
        self.timer = self.create_timer(1/30.0, self.timer_callback) # 约30FPS

    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 仅当提供了序列号时才启用设备
        if self.serial_number:
            config.enable_device(self.serial_number)
            
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as e:
            self.get_logger().error(f"节点 '{self.get_name()}' 无法启动 Realsense Pipeline (序列号: {self.serial_number}): {e}")
            raise e
            
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # 获取内参，注意：如果启用不同的设备，需要确保获取的是当前设备的内参
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[self.intr.fx, 0, self.intr.ppx],
                                       [0, self.intr.fy, self.intr.ppy],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(4)

    def init_mediapipe(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.8,
                                  min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def timer_callback(self):
        """代替 run_loop 中的一次循环迭代"""
        
        # 1. 获取帧
        try:
            # 使用 try_wait_for_frames 避免阻塞 timer
            frames = self.pipeline.wait_for_frames(timeout_ms=1000) 
            if not frames:
                return
        except Exception as e:
            self.get_logger().error(f"等待帧时出错: {e}")
            return
            
        # 2. 帧处理与对齐
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return
        # --- [关键修改 A] 保存原始数据 (未旋转前) 用于位姿计算 ---
        raw_color_image = np.asanyarray(color_frame.get_data()) 
        raw_depth_image = np.asanyarray(depth_frame.get_data()) 
        img_h, img_w = raw_color_image.shape[:2]

        # 3. 新增：腕部相机旋转逻辑
        color_image = np.asanyarray(color_frame.get_data()) # 将原始数据转为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data()) # 获取深度图的 numpy 数组，以便后续同步旋转

        # 仅针对腕部相机进行 180 度旋转
        is_wrist_cam = (self.camera_frame_id == 'wrist_cam_optical_frame')
        if is_wrist_cam:
            color_image = cv2.rotate(color_image, cv2.ROTATE_180)
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
        # -----------------------
        self.latest_image = color_image # 只需要将画好线和框的图片存起来，供主线程显示
        # 4. MediaPipe 处理（使用旋转后的图像）
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        # --- 位姿计算与发布 ---
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 可视化 (在另一个线程中运行时，尽量避免CV操作，但这里保留)
                self.mp_drawing.draw_landmarks(
                    image=color_image, landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

                # --- [关键修改 B] 坐标映射函数 ---
                # 将旋转图上的坐标点映射回原始图上的坐标点
                def get_orig_uv(landmark):
                    u_rot = landmark.x * img_w
                    v_rot = landmark.y * img_h
                    if is_wrist_cam:
                        # 180度映射逻辑: 原始 = 宽/高 - 旋转后
                        return (float(img_w - u_rot), float(img_h - v_rot))
                    return (float(u_rot), float(v_rot))
                
                # # --- 步骤 1: 使用 solvePnP 获取姿态 (Rotation) ---
                # image_points_2d = np.array([
                #     (int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h))
                #     for idx in self.model_points_indices
                # ], dtype=np.float64)
                # --- 步骤 1: 使用映射回来的【原始 2D 点】进行 solvePnP ---
                image_points_2d_orig = np.array([
                    get_orig_uv(face_landmarks.landmark[idx])
                    for idx in self.model_points_indices
                ], dtype=np.float64)

                (success, rotation_vector, tvec_for_vis) = cv2.solvePnP(
                    self.model_points_3d, image_points_2d_orig, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if not success:
                    continue
                    
                rotation_vector = np.array([rotation_vector[0], rotation_vector[1], rotation_vector[2]])
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # --- 步骤 2: 使用深度相机获取精确的位置 (Position) ---
                mouth_points_2d = [ (int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h)) for idx in self.mouth_indices ]
                mouth_center_2d = np.mean(mouth_points_2d, axis=0).astype(int)
                
                mouth_center_2d[0] = np.clip(mouth_center_2d[0], 0, img_w - 1)
                mouth_center_2d[1] = np.clip(mouth_center_2d[1], 0, img_h - 1)
                
                # depth = depth_frame.get_distance(mouth_center_2d[0], mouth_center_2d[1])
                # 改为这一行（从旋转后的数组取值，并从mm转换为m）：depth = depth_image[mouth_center_2d[1], mouth_center_2d[0]] * 0.001
                if is_wrist_cam:
                    # 180度映射逻辑: 原始 = 宽/高 - 旋转后
                    u = int(round(img_w - mouth_center_2d[0]))
                    v = int(round(img_h - mouth_center_2d[1]))
                else:
                    u = int(round(mouth_center_2d[0]))
                    v = int(round(mouth_center_2d[1]))
                # 1. 确保计算出来的 2D 坐标是整数

                # 2. 将坐标强制限制在相机内参允许的范围内
                u = max(0, min(u, self.intr.width - 1)) # RealSense 内参定义的范围通常是 [0, width-1] 和 [0, height-1]
                v = max(0, min(v, self.intr.height - 1)) # 注意：一定要使用 self.intr.width 和 self.intr.height，这是 SDK 最认可的边界

                # 3. 从未旋转的深度图中取值
                depth = raw_depth_image[v, u] * 0.001
                
                if depth > 0:
                    pos_in_camera_frame = rs.rs2_deproject_pixel_to_point(
                        self.intr, [float(u), float(v)], depth
                    )
                    
                    # -----------------------------------------------------------------
                    # 【计算并发布嘴唇垂直距离】（3D 物理距离，单位：米）
                    # 1. 计算像素坐标并强制限制边界 (使用 numpy clip)
                    # 【计算并发布嘴唇垂直距离】（同样映射回原始图计算）
                    def get_orig_uv(landmark_idx):
                        curr_u = face_landmarks.landmark[landmark_idx].x * img_w
                        curr_v = face_landmarks.landmark[landmark_idx].y * img_h
                        if is_wrist_cam:
                            return int(img_w - curr_u), int(img_h - curr_v)
                        return int(curr_u), int(curr_v)

                    u_top, v_top = get_orig_uv(self.TOP_LIP_IDX)
                    u_bot, v_bot = get_orig_uv(self.BOTTOM_LIP_IDX)

                    # 边界检查
                    u_top, v_top = np.clip(u_top, 0, img_w-1), np.clip(v_top, 0, img_h-1)
                    u_bot, v_bot = np.clip(u_bot, 0, img_w-1), np.clip(v_bot, 0, img_h-1)

                    # 2. 从原始深度图取深度
                    d_top = raw_depth_image[v_top, u_top] * 0.001
                    d_bottom = raw_depth_image[v_bot, u_bot] * 0.001

                    if d_top > 0 and d_bottom > 0:
                        # 3. 反投影到 3D 空间
                        # 确保传入的是 float 列表且坐标在合法范围内
                        pos_top = rs.rs2_deproject_pixel_to_point(self.intr, [float(u_top), float(v_top)], d_top)
                        pos_bottom = rs.rs2_deproject_pixel_to_point(self.intr, [float(u_bot), float(v_bot)], d_bottom)

                        # 计算 3D 欧氏距离 (米)
                        vertical_dist_3d = np.linalg.norm(np.array(pos_top) - np.array(pos_bottom))

                        mouth_msg = Float32()
                        mouth_msg.data = vertical_dist_3d # 现在是 3D 物理距离 (米)
                        self.mouth_dist_publisher_.publish(mouth_msg)
                        self.get_logger().info(f"[{self.get_name()}] Mouth Dist (3D): {vertical_dist_3d:.4f} m")
                    # -----------------------------------------------------------------
                    
                    # --- 步骤 3: 组合位姿，并以 PoseStamped 格式发布 ---
                    rotation_vector_final, _ = cv2.Rodrigues(rotation_matrix)
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    
                    pose_msg.header.frame_id = self.camera_frame_id 
                    
                    pose_msg.pose.position.x = pos_in_camera_frame[0]
                    pose_msg.pose.position.y = pos_in_camera_frame[1]
                    pose_msg.pose.position.z = pos_in_camera_frame[2]
                    
                    r = R.from_rotvec(rotation_vector_final.flatten())
                    quat = r.as_quat() # [x, y, z, w]
                    
                    pose_msg.pose.orientation.x = quat[0]
                    pose_msg.pose.orientation.y = quat[1]
                    pose_msg.pose.orientation.z = quat[2]
                    pose_msg.pose.orientation.w = quat[3]

                    # --- 步骤 4: 发布最终的位姿指令 ---
                    self.publisher_.publish(pose_msg)
                    self.get_logger().info(
                        f'[{self.get_name()}] Pos: ({pos_in_camera_frame[0]:.2f}, {pos_in_camera_frame[1]:.2f}, {pos_in_camera_frame[2]:.2f})'
                    )
                    

                # --- 可视化部分 ---
                mouth_3d_coords_cm = np.array(pos_in_camera_frame) * 100 if depth > 0 else None
                # --- 步骤 5: 可视化修正 ---
                if is_wrist_cam:
                    # 如果旋转了 180 度，我们要用旋转后的 2D 点重新算一个用于显示的位姿(仅用于可视化)
                    # 这样 projectPoints 算出来的坐标才能直接画在旋转图上
                    image_points_2d_vis = np.array([
                        (int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h))
                        for idx in self.model_points_indices
                    ], dtype=np.float64)
                    
                    _, rvec_vis, tvec_vis = cv2.solvePnP(
                        self.model_points_3d, image_points_2d_vis, self.camera_matrix, self.dist_coeffs
                    )
                else:
                    rvec_vis, tvec_vis = rotation_vector, tvec_for_vis

                # 调用可视化（使用修正后的 vis 参数）
                self.visualize_all_info(color_image, rvec_vis, tvec_vis,
                                        mouth_center_2d, mouth_3d_coords_cm)
        
        

    def visualize_all_info(self, image, rvec, tvec, mouth_center_2d, mouth_3d_coords_cm):
        """在图像上绘制所有需要的信息"""
        # (可视化代码保持不变，省略以节省空间)
        axis_points_3d = np.array([(0,0,0), (5,0,0), (0,5,0), (0,0,5)], dtype=np.float64)
        axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axis_points_2d = axis_points_2d.astype(int).reshape(-1, 2)
        cv2.line(image, axis_points_2d[0], axis_points_2d[1], (255,0,0), 3)   
        cv2.line(image, axis_points_2d[0], axis_points_2d[2], (0,255,0), 3)
        cv2.line(image, axis_points_2d[0], axis_points_2d[3], (0,0,255), 3)

        cv2.circle(image, tuple(mouth_center_2d), 5, (0, 0, 255), -1)

        mouth_text = f"Mouth Pos (cm): {np.array2string(mouth_3d_coords_cm, precision=1)}" \
                     if mouth_3d_coords_cm is not None else "Mouth Pos (cm): N/A"
        cv2.putText(image, mouth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        cv2.putText(image, f"Pitch: {angles[0]:.1f}, Yaw: {angles[1]:.1f}, Roll: {angles[2]:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        base_y = 90
        cv2.putText(image, "R Matrix:", (10, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i in range(3):
            row_str = f"[{rmat[i, 0]:.2f}, {rmat[i, 1]:.2f}, {rmat[i, 2]:.2f}]"
            cv2.putText(image, row_str, (10, base_y + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    def cleanup(self):
        self.timer.cancel()
        self.pipeline.stop()
        self.face_mesh.close()
        # cv2.destroyAllWindows() # 在主程序结束后统一清理
        self.get_logger().info(f"节点 '{self.get_name()}' 资源已清理。")

def main(args=None):
    rclpy.init(args=args)
    # ----------------------------------------------------
    WRIST_CAM_SN = "043322071261"  # 腕部相机序列号
    FIXED_CAM_SN = "244622070977"  # 固定相机序列号
    # ----------------------------------------------------

    # 1. 实例化两个相机节点
    wrist_node = FacePosePublisher(
        node_name='wrist_face_pose_node',
        camera_frame_id='wrist_cam_optical_frame',
        publish_topic='/wrist_face_pose',
        serial_number=WRIST_CAM_SN
    )
    
    fixed_node = FacePosePublisher(
        node_name='fixed_face_pose_node',
        camera_frame_id='fixed_cam_optical_frame',
        publish_topic='/fixed_face_pose',
        serial_number=FIXED_CAM_SN
    )
    
    # 2. 创建多线程执行器 (MultiThreadedExecutor)
    # Realsense I/O 和 MediaPipe 处理都是计算密集型和 I/O 密集型操作，
    # 必须使用多线程来确保两个节点都能及时处理数据。
    executor = MultiThreadedExecutor()
    executor.add_node(wrist_node)
    executor.add_node(fixed_node)
    
    # try:
    #     executor.spin() # 运行所有节点

    try:
        # 【关键修改 B】手动接管主循环
        while rclpy.ok():
            # 让 ROS 执行器运行 10 毫秒，处理回调（计算位姿并发布）
            executor.spin_once(timeout_sec=0.01)

            # 在主线程中刷新两个窗口
            if wrist_node.latest_image is not None:
                cv2.imshow('Wrist Camera (MediaPipe)', wrist_node.latest_image)
            
            if fixed_node.latest_image is not None:
                cv2.imshow('Fixed Camera (MediaPipe)', fixed_node.latest_image)

            # 统一处理按键事件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        wrist_node.get_logger().error(f"执行器发生异常: {e}")
    finally:
        # 3. 清理资源
        wrist_node.cleanup()
        fixed_node.cleanup()
        cv2.destroyAllWindows()
        wrist_node.destroy_node()
        fixed_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()