import cv2
import face_recognition
import os

# Step 1: Train known faces from a folder
path = 'faces'  # Folder name where known faces are stored
images = []
classNames = []

# সব face image load করা
for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

# Function: image থেকে encoding বের করা
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("[INFO] Training faces...")
encodeListKnown = findEncodings(images)
print("Training Complete ✅")

# Step 2: Start webcam
cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = min(range(len(faceDis)), key=faceDis.__getitem__)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = "UNKNOWN"

        y1, x2, y2, x1 = [v * 4 for v in faceLoc]  # scale up (0.25 → 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, name, (x1 + 6, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cam.release()
cv2.destroyAllWindows()
