import cv2

openCapture = cv2.VideoCapture(0)

while True:
    _, frame = openCapture.read()
    cv2.imshow('Video', frame)

    

    if cv2.waitKey(1) & 0xFF == ord('q'): #if input it q
        break

openCapture.release()
cv2.destroyAllWindows()