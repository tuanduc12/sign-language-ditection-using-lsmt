import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from PIL import Image, ImageDraw, ImageFont

# Khởi tạo Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Hàm phát hiện Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

# Vẽ các điểm mốc với phong cách đẹp hơn
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(255, 182, 193), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(240, 248, 255), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 165, 0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 215, 0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 191, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(135, 206, 235), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 69, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 99, 71), thickness=2, circle_radius=2))

# Trích xuất keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose, face, lh, rh])

# Hàm vẽ văn bản tiếng Việt bằng Pillow
def draw_text(image, text, position, font_path, font_size, color):
    # Chuyển từ OpenCV (BGR) sang Pillow (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Tải font hỗ trợ tiếng Việt (Arial hoặc font khác có sẵn trên máy)
    font = ImageFont.truetype(font_path, font_size)
    
    # Vẽ văn bản
    draw.text(position, text, font=font, fill=color)
    
    # Chuyển lại sang định dạng OpenCV (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Tải danh sách hành động
if os.path.exists('actions.npy'):
    actions = np.load('actions.npy')
else:
    actions = np.array(['null', 'xin chào', 'cảm ơn', 'xin lỗi', 'hạnh phúc', 'tuyệt vời', 'yêu thương', 'ghét', 'biết ơn', 'tạm biệt'])

# Tải mô hình
model = load_model('action.h5')

# Khởi tạo biến
sequence = []
current_sentence = ""
threshold = 0.8
last_prediction_time = 0
prediction_interval = 0.5

# Đường dẫn tới font hỗ trợ tiếng Việt (thay đổi nếu cần)
font_path = "C:/Windows/Fonts/arial.ttf"  # Font Arial mặc định trên Windows

# Mở camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Tối ưu hóa với Mediapipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý ở độ phân giải 960x540
        image = cv2.resize(frame, (960, 540))
        image, results = mediapipe_detection(image, holistic)
        draw_styled_landmarks(image, results)

        # Trích xuất keypoints
        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]

        # Dự đoán hành động
        current_time = time.time()
        if len(sequence) == 30 and (current_time - last_prediction_time) >= prediction_interval:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold:
                current_sentence = actions[np.argmax(res)]
            else:
                current_sentence = "Đang nhận diện..."
            last_prediction_time = current_time

        # Vẽ giao diện với tiếng Việt
        # Thanh tiêu đề
        cv2.rectangle(image, (0, 0), (960, 60), (50, 50, 50), -1)
        image = draw_text(image, "Nhận Diện Ngôn Ngữ Ký Hiệu", (20, 15), font_path, 35, (255, 255, 255))

        # Thanh kết quả
        cv2.rectangle(image, (0, 540-50), (960, 540), (50, 50, 50), -1)
        image = draw_text(image, f"Hành động: {current_sentence}", (20, 540-40), font_path, 30, (0, 255, 0))

        # Hiển thị khung hình
        cv2.imshow('Nhận diện ngôn ngữ ký hiệu', image)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()