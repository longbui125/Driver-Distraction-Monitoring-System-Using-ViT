import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os
from collections import deque
from statistics import mode

from vit_model import ViT_Model
from vit_keras import vit

# --- KHỞI TẠO MODEL ---
model = ViT_Model(input_shape=(224, 224, 3), classes=10)
model.load_weights('vit_distraction_weights.weights.h5')
print("Model weights loaded successfully!")

# --- BIẾN TOÀN CỤC ---
video_path = None
running = False
cap = None
out = None 

CONFIDENCE_THRESHOLD = 0.7
HISTORY_LENGTH = 15 # Số lượng frame để tính trung bình trượt

# Hàng đợi lưu trữ lịch sử dự đoán để chống nhảy nhãn
prediction_history = deque(maxlen=HISTORY_LENGTH)

classes = ["safe driving", "texting - right", "talking on the phone - right", 
           "texting - left", "talking on the phone - left", "operating the radio", 
           "drinking", "reaching behind", "hair and makeup", "talking to passenger"]

def choose_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if video_path:
        file_label.config(text=f"Selected: {video_path}")

def show_frame():
    global running, cap, out
    if not running or cap is None: return
    
    ret, frame = cap.read()
    if not ret:
        stop_monitoring()
        return

    # ---------------------------------------------------------
    # 1. CẮT KHUNG HÌNH (CROP ROI)
    # Bỏ khoảng 30% không gian bên phải (ghế phụ) để model tập trung vào tài xế
    # Giúp giảm lỗi nhầm "Safe" thành "Talking to passenger"
    # ---------------------------------------------------------
    height, width, _ = frame.shape
    crop_width = int(width * 0.9) # Chỉ lấy 70% chiều rộng từ trái sang
    cropped_frame = frame[:, :crop_width] 

    # Chuyển đổi màu và resize từ ảnh ĐÃ CẮT
    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(frame_rgb, (224, 224))
    input_data = vit.preprocess_inputs(input_data)
    input_data = np.expand_dims(input_data, axis=0)

    # ---------------------------------------------------------
    # 2. DỰ ĐOÁN VÀ LÀM MƯỢT (TEMPORAL SMOOTHING)
    # ---------------------------------------------------------
    prediction = model.predict(input_data, verbose=0)
    idx = np.argmax(prediction[0])
    
    # Thêm nhãn dự đoán được vào lịch sử
    prediction_history.append(idx)
    
    # Tìm nhãn xuất hiện nhiều nhất trong N frames gần nhất
    try:
        stable_idx = mode(prediction_history)
    except:
        # Trong trường hợp có 2 nhãn xuất hiện bằng nhau, lấy nhãn hiện tại
        stable_idx = idx 
        
    stable_confidence = prediction[0][stable_idx]

    # --- Xử lý hiển thị ---
    if stable_confidence >= CONFIDENCE_THRESHOLD:
        predicted_class = classes[stable_idx]
        text_color = (0, 255, 0) # Xanh lá cho độ tin cậy cao
    else:
        predicted_class = classes[stable_idx]
        text_color = (0, 255, 255) # Vàng cho độ tin cậy thấp

    # Resize frame gốc để hiển thị (vẫn giữ nguyên video gốc để xem đầy đủ)
    display_frame = cv2.resize(frame, (800, 450))
    
    # Vẽ hộp thông báo để chữ dễ đọc hơn trên nền video
    cv2.rectangle(display_frame, (20, 10), (600, 60), (0, 0, 0), -1)
    cv2.putText(display_frame, f"{predicted_class} ({stable_confidence*100:.1f}%)", 
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    if out is not None:
        out.write(display_frame)

    display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    root.after(10, show_frame)

def start_monitoring():
    global running, cap, out, video_path
    if video_path:
        prediction_history.clear() # Xóa lịch sử cũ khi bắt đầu video mới
        running = True
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
        
        dir_name, file_name = os.path.split(video_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(dir_name, f"{name}_output.avi")
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (800, 450))
        print(f"[INFO] Bắt đầu ghi video tại: {output_path}")

        show_frame()

def stop_monitoring():
    global running, cap, out
    running = False
    if cap: cap.release()
    
    if out is not None:
        out.release()
        out = None
        print("[INFO] Đã lưu xong video output!")
    
    video_label.config(image="")

# --- GIAO DIỆN Tkinter ---
root = Tk()
root.title("Driver Distraction Monitoring System - Stabilized")
root.geometry("1024x700")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)
monitor_tab = Frame(notebook)
notebook.add(monitor_tab, text="Monitoring")

file_frame = Frame(monitor_tab); file_frame.pack(fill="x", pady=5)
file_label = Label(file_frame, text="No file selected", font=("Arial", 12)); file_label.pack(side="left", padx=10)
Button(file_frame, text="Choose File", command=choose_file, bg="blue", fg="white").pack(side="right", padx=10)

video_frame = Frame(monitor_tab, bg="black"); video_frame.pack(fill="both", expand=True, padx=10, pady=10)
video_label = Label(video_frame, bg="black"); video_label.pack(fill="both", expand=True)

button_frame = Frame(monitor_tab); button_frame.pack(pady=20)
Button(button_frame, text="Start", command=start_monitoring, bg="green", fg="white", width=10).pack(side="left", padx=10)
Button(button_frame, text="Stop", command=stop_monitoring, bg="red", fg="white", width=10).pack(side="left", padx=10)

root.mainloop()