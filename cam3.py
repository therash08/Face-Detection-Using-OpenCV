import cv2
import mediapipe as mp
import pyautogui

# MediaPipe hand detection setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam চালু করা
cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # মিরর ইফেক্ট (ডান হাত ঠিকমতো দেখা যায়)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # RGB তে রূপান্তর
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # যদি হাত detect হয়
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Index finger এর landmark
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)

                # Cursor move করানো
                screen_x = int(hand_landmarks.landmark[8].x * screen_w)
                screen_y = int(hand_landmarks.landmark[8].y * screen_h)
                pyautogui.moveTo(screen_x, screen_y)

                # যদি thumb ও index কাছাকাছি আসে → click
                x_thumb = int(hand_landmarks.landmark[4].x * w)
                y_thumb = int(hand_landmarks.landmark[4].y * h)
                distance = abs(x - x_thumb) + abs(y - y_thumb)
                if distance < 40:
                    pyautogui.click()
                    cv2.putText(frame, 'CLICK!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Hand Cursor Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
