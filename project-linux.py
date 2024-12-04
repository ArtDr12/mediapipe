import numpy as np
import mediapipe as mp
import cv2
import uinput # на линуксе keyboard требует права администратора, нагуглил эту библиотеку

def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)

def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2)**2 + (y1 - y2) **2) **.5

#создаем детектор
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
x_wrist_old = y_wrist_old = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    # Распознаем
    results = handsDetector.process(flippedRGB)
    # Рисуем распознанное, если распозналось
    if results.multi_hand_landmarks is not None:
        # Находим запястье, обозначим красной точкой, если кулак не зажат, зелёной,если зажат
        x_wrist = int(results.multi_hand_landmarks[0].landmark[0].x * flippedRGB.shape[1])
        y_wrist = int(results.multi_hand_landmarks[0].landmark[0].y * flippedRGB.shape[0])
        
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        
        if 2 * r / ws < 1.5:
            cv2.circle(flippedRGB, (x_wrist, y_wrist), 10, (0, 255, 0), -1)
            cv2.line(flippedRGB, (x_wrist, y_wrist), (x_wrist_old, y_wrist_old), (0, 255, 0), 2)
            if x_wrist - x_wrist_old > 50: # Определяется движение рукой если рука в кулаке
                print('вправо') 
                with uinput.Device([uinput.KEY_RIGHT]) as device:
                    device.emit_click(uinput.KEY_RIGHT) 
            elif x_wrist - x_wrist_old < -50:
                print('влево')
                with uinput.Device([uinput.KEY_LEFT]) as device:
                    device.emit_click(uinput.KEY_LEFT)
        else:
            cv2.circle(flippedRGB, (x_wrist, y_wrist), 10, (255, 0, 0), -1)
            cv2.line(flippedRGB, (x_wrist, y_wrist), (x_wrist_old, y_wrist_old), (255, 0, 0), 2)

        x_wrist_old, y_wrist_old = x_wrist, y_wrist
    # Gереводим в BGR и показываем результат
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

handsDetector.close()
