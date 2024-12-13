import cv2
import numpy as np
import pandas as pd
import dlib
import os
import matplotlib.pyplot as plt

# 配置區
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(DLIB_LANDMARK_PATH):
    raise FileNotFoundError("請先確保 dlib 特徵點模型檔案存在: " + DLIB_LANDMARK_PATH)

dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)

cheek_landmarks_indices_dlib = [2,3,4,5,6,7,8,9,10,11,12,13,14]

# 通用函式
def _compute_ippg_from_roi(frame, roi_points):
    """計算ROI內的R/G/B平均值"""
    if len(roi_points) == 0:
        return None, None, None
    xs = [p[0] for p in roi_points]
    ys = [p[1] for p in roi_points]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    h, w, _ = frame.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w-1, x_max)
    y_max = min(h-1, y_max)

    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None, None

    mean_color = np.mean(roi.reshape(-1,3), axis=0)
    R, G, B = mean_color[2], mean_color[1], mean_color[0]  # OpenCV: BGR
    return R, G, B

# 提取 ROI 並返回數據與第一幀的示意圖
def extract_roi_dlib_with_visualization(video_file):
    """使用 dlib 提取 ROI 並生成示意圖與數據"""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError("無法開啟影片: " + video_file)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    time_list = []
    R_list = []
    G_list = []
    B_list = []

    frame_idx = 0
    roi_visualization = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dlib_detector(gray, 0)
        if len(faces) > 0:
            face = faces[0]
            shape = dlib_predictor(gray, face)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            
            cheek_points = [landmarks[i] for i in cheek_landmarks_indices_dlib]
            R, G, B = _compute_ippg_from_roi(frame, cheek_points)
            if R is not None:
                time_list.append(frame_idx / fps)
                R_list.append(R)
                G_list.append(G)
                B_list.append(B)

            # 在第一幀生成示意圖
            if frame_idx == 0:
                for point in cheek_points:
                    cv2.circle(frame, point, 3, (0, 255, 0), -1)
                roi_visualization = frame

        frame_idx += 1

    cap.release()
    df = pd.DataFrame({
        'time': time_list,
        'R': R_list,
        'G': G_list,
        'B': B_list
    })
    return df, roi_visualization
