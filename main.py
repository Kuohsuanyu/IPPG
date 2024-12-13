import cv2
import matplotlib.pyplot as plt
from roi_alt import roi_dilb
from tkinter import filedialog
import os

def main():
    # 設定影片路徑
    video_file = filedialog.askopenfilename(
        title="選擇影片文件",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    if not video_file:
        print("未選擇影片文件，程式結束。")
        return

    try:
        # 抓取 ROI 和生成示意圖
        df, roi_image = roi_dilb.extract_roi_dlib_with_visualization(video_file)
        
        # 顯示 ROI 示意圖
        plt.figure("ROI 定位示意圖")
        plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        plt.title("ROI 定位示意圖")
        plt.axis("off")
        plt.show()

        # 繪製 iPPG 信號圖
        plt.figure("iPPG 信號圖")
        plt.plot(df['time'], df['R'], label="R 通道", color="red")
        plt.plot(df['time'], df['G'], label="G 通道", color="green")
        plt.plot(df['time'], df['B'], label="B 通道", color="blue")
        plt.xlabel("時間 (秒)")
        plt.ylabel("像素值")
        plt.legend()
        plt.title("三通道 iPPG 信號")
        plt.show()

        # 是否儲存結果
        save = input("是否儲存結果？(y/n): ").strip().lower()
        if save == "y":
            save_path = filedialog.asksaveasfilename(
                title="選擇儲存位置",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if save_path:
                df.to_csv(save_path, index=False)
                print(f"結果已儲存到：{save_path}")
            else:
                print("未選擇儲存位置，未儲存結果。")
        else:
            print("未儲存結果。")

    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
