import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(1)
    if key == ord('s'): 
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(f"face_{count}.jpg", face_img)
            print(f"Saved face_{count}.jpg")
            count += 1
    elif key == ord('r'):
        break

cam.release()
cv2.destroyAllWindows()
