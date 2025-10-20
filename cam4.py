import cv2
import winsound   
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
count = 0
last_beep_time = 0 

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_count = len(faces)  
    if face_count > 0 and time.time() - last_beep_time > 1:
        winsound.Beep(700, 200)  # (frequency, duration)
        last_beep_time = time.time()


    for (x, y, w, h) in faces:
       
        name = "Unknown"
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)
        
       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

   
    cv2.putText(frame, f"Faces Detected: {face_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow('Advanced Face Detection', frame)

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
