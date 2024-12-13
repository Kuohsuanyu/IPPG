# roi_extractors.py
import cv2
import numpy as np
import pandas as pd
import dlib
import mediapipe as mp
import os

#######################################
# 配置區：請根據實際需求修改
#######################################

# dlib 模型路徑(請確保該檔案存在)
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(DLIB_LANDMARK_PATH):
    raise FileNotFoundError("請先確保 dlib 特徵點模型檔案存在: " + DLIB_LANDMARK_PATH)

# 初始化 dlib 人臉偵測與臉部特徵點預測器
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)

# 初始化 MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 定義臉頰區域的 landmark 索引(僅為示意，請依實際標準)
# dlib 68 點模型中，臉頰常為臉部輪廓線的一部分(如0-16為臉部外輪廓)
# 可根據需要選擇某段，如(2,3,4,5,...9)代表左臉頰區域
cheek_landmarks_indices_dlib = [2,3,4,5,6,7,8,9,10,11,12,13,14]

# MediaPipe FaceMesh 468點標記中臉頰區域的索引(需自行查參考圖，這裡用假設值)
# 假設臉頰區域為某一組點集，如 [50,52,58,234,230]等，請依實際情況修改
cheek_landmarks_indices_mp = [50,52,58,234,230,226,205,202,210]

#######################################
# 通用函式
#######################################

def _compute_ippg_from_roi(frame, roi_points):
    """
    給定一張彩色圖(frame)與ROI的點集，計算該ROI範圍內的R/G/B平均值。
    簡化做法：使用包含所有點的最小外接矩形區域作為ROI。
    實務中可細化為使用多邊形Mask取得更精確的ROI。
    """
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

#######################################
# dlib 68點方法
#######################################
def extract_roi_dlib(video_file):
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
            
            # 取臉頰區域點
            cheek_points = [landmarks[i] for i in cheek_landmarks_indices_dlib]

            R, G, B = _compute_ippg_from_roi(frame, cheek_points)
            if R is not None:
                time_list.append(frame_idx/fps)
                R_list.append(R)
                G_list.append(G)
                B_list.append(B)

        frame_idx += 1

    cap.release()
    df = pd.DataFrame({
        'time': time_list,
        'R': R_list,
        'G': G_list,
        'B': B_list
    })
    return df

#######################################
# MediaPipe FaceMesh方法
#######################################
def extract_roi_googlesearch458(video_file):
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
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                    # 取臉頰區域點
                    cheek_points = [landmarks[i] for i in cheek_landmarks_indices_mp if i < len(landmarks)]
                    
                    R, G, B = _compute_ippg_from_roi(frame, cheek_points)
                    if R is not None:
                        time_list.append(frame_idx/fps)
                        R_list.append(R)
                        G_list.append(G)
                        B_list.append(B)

            frame_idx += 1

    cap.release()
    df = pd.DataFrame({
        'time': time_list,
        'R': R_list,
        'G': G_list,
        'B': B_list
    })
    return df
