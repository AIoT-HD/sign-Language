import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
max_num_hands = 1
gesture = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ',
    6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ',
    12: 'ㅍ', 13: 'ㅎ', 
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ',
    20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 
    26: 'ㅔ', 27: 'ㅖ', 28: 'ㅢ', 29: 'ㅚ', 30: 'ㅟ'
}

# MediaPipe 손 모델 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)



# 제스처 인식 모델

# 현재 스크립트의 디렉토리 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
# 상대 경로로 파일 경로 설정
data_file_path = os.path.join(current_dir, 'data/gesture_train_kr.csv')

file = np.genfromtxt(data_file_path, delimiter=',')
angle = file[:, :-2].astype(np.float32)  # 라벨과 카운트를 제외한 각도 데이터
label = file[:, -2].astype(np.float32)  # 두 번째 마지막 열은 라벨
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# TrueType 폰트 로드 (시스템의 폰트 경로에 맞게 수정)
font_path = "C:/Windows/Fonts/gulim.ttc"  # 시스템에 맞게 폰트 경로 수정
font = ImageFont.truetype(font_path, 48)  # 글꼴 크기를 크게 조정

cap = cv2.VideoCapture(0)

def draw_text_with_pillow(image, text, position, font, font_color=(255, 255, 255)):
    """Pillow를 사용하여 이미지에 텍스트를 그리는 함수"""
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=font_color)
    return np.array(image_pil)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # 관절 사이의 각도 계산
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # 부모 관절
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # 자식 관절
            v = v2 - v1  # [20,3]
            # v 정규화
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 벡터의 내적을 사용하여 각도 계산
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # 라디안에서 각도로 변환

            # 제스처 예측
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # 제스처 결과 표시 (한국어)
            if idx in gesture.keys():
                gesture_text = gesture[idx]
                position = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20))
                img = draw_text_with_pillow(img, gesture_text, position, font)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('수어 인식', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()