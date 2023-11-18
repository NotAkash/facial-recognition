import cv2
import face_recognition

openCapture = cv2.VideoCapture(0)
openCapture.set(3, 400)  # Width
openCapture.set(4, 400)  # Height

skipF = 5  # Adjust as needed
frameCount = 0

# Explicitly set the window property
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Move this line after 'cv2.namedWindow' to ensure it takes effect
cv2.resizeWindow("Video", 400, 400)

while True:
    _, frame = openCapture.read()

    if frameCount % skipF == 0:
        detectFace = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in detectFace:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # if input is 'q'
        break

    frameCount += 1

openCapture.release()
cv2.destroyAllWindows()
