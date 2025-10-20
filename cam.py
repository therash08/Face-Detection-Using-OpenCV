# import cv2
# cam = cv2.VideoCapture(2)
# while True:
#     _, frame = cam.read()
#     CV2.imshow('my webcam', frame)
#     cv2.waitKey(1)

import cv2

# 0 সাধারণত ডিফল্ট ক্যামেরা (ল্যাপটপের বিল্ট-ইন)
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow('My Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

