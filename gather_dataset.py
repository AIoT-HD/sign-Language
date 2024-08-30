import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

max_num_hands = 1
gesture = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ',
    6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ',
    12: 'ㅍ', 13: 'ㅎ', 
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ',
    20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 
    26: 'ㅔ', 27: 'ㅖ', 28: 'ㅢ', 29: 'ㅚ', 30: 'ㅟ'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)



# 현재 스크립트의 디렉토리 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
# 상대 경로로 파일 경로 설정
data_path = os.path.join(current_dir, 'data/gesture_train_kr.csv')

# Check if the data file exists, otherwise initialize it
if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
    file = np.genfromtxt(data_path, delimiter=',')
    counts = {i: np.sum(file[:, -2] == i) for i in range(len(gesture))}
else:
    file = np.empty((0, 17))  # Initialize with an empty array with the correct shape
    counts = {i: 0 for i in range(len(gesture))}

print(file.shape)

cap = cv2.VideoCapture(0)

# Initialize the current gesture label
current_label = 0

# Load the TrueType font
font_path = "C:/Windows/Fonts/gulim.ttc"  # Update to the correct path on your system
font = ImageFont.truetype(font_path, 48)

def click(event, x, y, flags, param):
    global data, file, counts
    if event == cv2.EVENT_LBUTTONDOWN:
        counts[current_label] += 1
        data_with_count = np.append(data, [current_label, counts[current_label]])
        file = np.vstack((file, data_with_count))
        print(f"Data collected: {file.shape}, Gesture: {gesture[current_label]}, Count: {counts[current_label]}")

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

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

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arccos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Prepare the data array with the current label
            data = np.array([angle], dtype=np.float32)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    # Convert image to PIL for text rendering
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 30), f"Current Label: {gesture[current_label]}", font=font, fill=(0, 255, 0, 0))

    # Convert back to OpenCV format
    img = np.array(img_pil)

    cv2.imshow('Dataset', img)

    key = cv2.waitKey(1)
    
    if key == ord('`'):  # Press 'q' to exit
        break
    elif key == ord('r'):  # Press '0' for ㄱ
        current_label = 0
    elif key == ord('s'):  # Press '1' for ㄴ
        current_label = 1
    elif key == ord('e'):  # Press '2' for ㄷ
        current_label = 2
    elif key == ord('f'):  # Press '3' for ㄹ
        current_label = 3
    elif key == ord('a'):  # Press '4' for ㅁ
        current_label = 4
    elif key == ord('q'):  # Press '5' for ㅂ
        current_label = 5
    elif key == ord('t'):  # Press '6' for ㅅ
        current_label = 6
    elif key == ord('d'):  # Press '7' for ㅇ
        current_label = 7
    elif key == ord('w'):  # Press '8' for ㅈ
        current_label = 8
    elif key == ord('c'):  # Press '9' for ㅊ
        current_label = 9
    elif key == ord('z'):  # Press 'k' for ㅋ
        current_label = 10
    elif key == ord('x'):  # Press 't' for ㅌ
        current_label = 11
    elif key == ord('v'):  # Press 'p' for ㅍ
        current_label = 12
    elif key == ord('g'):  # Press 'h' for ㅎ
        current_label = 13
#모음   
    elif key == ord('k'):  # Press 'k' for ㅏ
        current_label = 14
    elif key == ord('i'):  # Press 'i' for ㅑ
        current_label = 15
    elif key == ord('j'):  # Press 'h' for ㅓ
        current_label = 16
    elif key == ord('u'):  # Press 'h' for ㅕ
        current_label = 17
    elif key == ord('h'):  # Press 'h' for ㅗ
        current_label = 18
    elif key == ord('y'):  # Press 'h' for ㅛ
        current_label = 19        
    elif key == ord('n'):  # Press 'h' for ㅜ
        current_label = 20
    elif key == ord('b'):  # Press 'h' for ㅠ
        current_label = 21
    elif key == ord('m'):  # Press 'h' for ㅡ
        current_label = 22
    elif key == ord('l'):  # Press 'h' for ㅣ
        current_label = 23
    elif key == ord('o'):  # Press 'h' for ㅐ
        current_label = 24
    elif key == ord('O'):  # Press 'h' for ㅒ
        current_label = 25
    elif key == ord('p'):  # Press 'h' for ㅔ
        current_label = 26
    elif key == ord('P'):  # Press 'h' for ㅖ
        current_label = 27
    elif key == ord('M'):  # Press 'h' for ㅢ
        current_label = 28
    elif key == ord('H'):  # Press 'h' for ㅚ
        current_label = 29        
    elif key == ord('N'):  # Press 'h' for ㅟ
        current_label = 30

# Save the collected data to a file
np.savetxt(data_path, file, delimiter=',')