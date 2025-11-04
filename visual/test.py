import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
# a. static_image_mode=True: 设置为静态图片模式。
# b. max_num_faces=1: 设置最多检测1张人脸。
# c. min_detection_confidence=0.5: 人脸检测的最小置信度。
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3, min_detection_confidence=0.5)

# 初始化 MediaPipe 绘图工具
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- 请在这里修改您的图片路径 ---
image_path = 'face/9.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"错误：无法加载图片，请检查路径是否正确: {image_path}")
else:
    # --- 新增的代码块：调整图片大小 ---
    # 获取原始图片的高度和宽度
    h, w, _ = image.shape
    # 设置一个目标宽度
    target_width = 800
    # 根据目标宽度，按比例计算新的高度
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    # 调整图片大小
    image = cv2.resize(image, (target_width, target_height))
    # --- 新增代码块结束 ---

    # MediaPipe 处理的是 RGB 图像，而 OpenCV 加载的是 BGR 格式，因此需要转换
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理图像，检测人脸特征点
    results = face_mesh.process(image_rgb)

    # 创建一个图片的副本，用于绘制结果
    annotated_image = image.copy()

    # MediaPipe Face Mesh 模型定义的嘴唇特征点索引
    # 这些索引是固定的，对应嘴唇轮廓
    LIPS_INDICES = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,  # 上外唇
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,  # 下外唇
        76, 77, 90, 180, 85, 16, 315, 404, 320, 375, 291,  # 上内唇
        191, 80, 81, 82, 13, 312, 311, 310, 415, 308,      # 下内唇
    ]
    # 将索引列表去重并排序
    unique_lips_indices = sorted(list(set(LIPS_INDICES)))


    # 如果检测到了人脸
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制整个人脸的网格
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION, # 绘制网格
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # 用不同颜色（红色）高亮嘴部特征点
            for index in unique_lips_indices:
                landmark = face_landmarks.landmark[index]
                # 特征点的坐标是归一化的 (0到1之间)，需要乘以图片的宽高转换成像素坐标
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                # 在嘴部特征点上画一个半径为2的红色实心圆
                cv2.circle(annotated_image, (x, y), 2, (0, 0, 255), -1)

    # 显示带有标注的图片
    cv2.imshow('Face and Mouth Landmark Detection', annotated_image)
    # 等待用户按键后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 释放资源
face_mesh.close()