import sys
import os
import re
import shutil
import pandas as pd
from datetime import datetime
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QFileDialog, QLabel, QMessageBox)
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("影片與ECG資料自動對齊並輸出資料夾")

        self.video_file = None
        self.ecg_file = None
        self.df_ecg = None
        self.ecg_start_time = None

        main_layout = QVBoxLayout()

        self.btn_load_video = QPushButton("選擇影片檔案")
        self.btn_load_video.clicked.connect(self.load_video)
        main_layout.addWidget(self.btn_load_video)

        self.btn_load_ecg = QPushButton("選擇ECG資料檔")
        self.btn_load_ecg.clicked.connect(self.load_ecg)
        main_layout.addWidget(self.btn_load_ecg)

        self.btn_align = QPushButton("開始對齊並儲存資料夾")
        self.btn_align.clicked.connect(self.align_and_save)
        main_layout.addWidget(self.btn_align)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇影片檔案", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_file = file_path
            QMessageBox.information(self, "影片載入", f"已選擇影片: {os.path.basename(file_path)}")

    def load_ecg(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇ECG資料檔案", "", "Data Files (*.csv *.xlsx)")
        if file_path:
            try:
                df = self.read_data_file(file_path)
                if 'time' in df.columns and 'ECG_signal' in df.columns and 'start_time' in df.columns:
                    self.ecg_file = file_path
                    self.df_ecg = df
                    ecg_start_str = df['start_time'].iloc[0]
                    self.ecg_start_time = self.parse_time_str(ecg_start_str)
                    if self.ecg_start_time is None:
                        QMessageBox.warning(self, "錯誤", f"ECG資料中 start_time 格式無法解析: {ecg_start_str}")
                        self.df_ecg = None
                        return
                    QMessageBox.information(self, "ECG載入成功", f"已選擇檔案: {os.path.basename(file_path)}\nECG開始時間:{self.ecg_start_time}")
                else:
                    QMessageBox.warning(self, "錯誤", "ECG資料中需要包含 time、ECG_signal 和 start_time 欄位")
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"讀取ECG檔案失敗:\n{e}")

    def align_and_save(self):
        if self.video_file is None or self.df_ecg is None or self.ecg_start_time is None:
            QMessageBox.warning(self, "警告", "請先匯入影片與ECG資料!")
            return

        try:
            video_start_time = self.extract_time_from_filename(self.video_file)
            if video_start_time is None:
                QMessageBox.warning(self, "警告", "無法從影片檔名解析時間，請確認檔名格式!")
                return

            aligned_df, time_shift = self.align_data(video_start_time, self.ecg_start_time, self.df_ecg)

        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"對齊過程中發生錯誤:\n{e}")
            return

        if aligned_df is not None and not aligned_df.empty:
            # 選擇儲存資料夾位置
            folder_path = QFileDialog.getExistingDirectory(self, "選擇儲存資料夾")
            if not folder_path:
                QMessageBox.information(self, "資訊", "已取消儲存")
                return

            # 建立結果資料夾，例如使用影片時間命名
            folder_name = f"aligned_{video_start_time.strftime('%Y%m%d_%H%M%S')}"
            result_dir = os.path.join(folder_path, folder_name)
            os.makedirs(result_dir, exist_ok=True)

            # 儲存對齊後資料
            aligned_file_path = os.path.join(result_dir, "aligned_data.csv")
            aligned_df.to_csv(aligned_file_path, index=False)

            # 複製原始ECG資料
            if self.ecg_file:
                shutil.copy(self.ecg_file, os.path.join(result_dir, "original_ecg_data" + os.path.splitext(self.ecg_file)[1]))

            # 複製原始影片
            shutil.copy(self.video_file, os.path.join(result_dir, "original_video" + os.path.splitext(self.video_file)[1]))

            # 產生對齊後影片(若需裁切)
            # 如果 time_shift < 0 表示 ECG 開始時間比影片晚，需從影片中略過前段
            # 若 time_shift >= 0表示ECG比影片早開始，現在對齊後的零秒即是原影片起點，可直接複製影片作為對齊後影片
            aligned_video_path = os.path.join(result_dir, "aligned_video.mp4")
            if time_shift < 0:
                # 從影片中略過 abs(time_shift) 秒
                self.create_aligned_video(self.video_file, aligned_video_path, start_time=abs(time_shift))
            else:
                # 不需裁切，直接複製原始影片作為對齊後影片
                shutil.copy(self.video_file, aligned_video_path)

            QMessageBox.information(self, "完成", f"所有檔案已儲存至資料夾: {result_dir}")
        else:
            QMessageBox.warning(self, "警告", "對齊後資料為空，無法儲存")

    def read_data_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.csv']:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df

    def extract_time_from_filename(self, file_path):
        """
        假設影片檔名格式為: test001_YYYY_MM_DD_HH_MM_SS.mp4
        """
        filename = os.path.basename(file_path)
        pattern = r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
        match = re.search(pattern, filename)
        if match:
            datetime_str = match.group(1)
            dt = datetime.strptime(datetime_str, "%Y_%m_%d_%H_%M_%S")
            return dt
        else:
            return None

    def parse_time_str(self, time_str):
        """
        假設ECG start_time格式為: 2024 18:40:49.089 (年 時:分:秒.毫秒)
        預設日期為1月1日
        """
        parts = time_str.split()
        if len(parts) < 2:
            return None

        year_str = parts[0]  # YYYY
        time_part = parts[1] # HH:MM:SS.fff
        new_time_str = f"{year_str}-01-01 {time_part}"
        try:
            dt = datetime.strptime(new_time_str, "%Y-%m-%d %H:%M:%S.%f")
            return dt
        except ValueError:
            return None

    def align_data(self, video_start_time, ecg_start_time, df_ecg):
        time_shift = (video_start_time - ecg_start_time).total_seconds()
        aligned_df = df_ecg.copy()
        aligned_df['time'] = aligned_df['time'] + time_shift
        aligned_df = aligned_df[aligned_df['time'] >= 0].reset_index(drop=True)
        return aligned_df, time_shift

    def create_aligned_video(self, input_video, output_video, start_time=0):
        """
        從 input_video 開始時間 start_time(秒) 處截取並保存為 output_video
        如果 start_time = 0，則直接複製即可，這裡示範進行簡單的影片裁切。
        """
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_time * fps)
        if start_frame < 0:
            start_frame = 0
        if start_frame >= frame_count:
            # 如果起始時間超過影片長度，就輸出空影片
            cap.release()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            out.release()
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
