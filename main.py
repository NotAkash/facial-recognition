import cv2
import face_recognition
import os

# Create a folder to save unique faces
output = "unique_faces"
os.makedirs(output, exist_ok=True)

# Load known faces from the "unique_faces" folder
knownFaces = []
knownFacesLabel = []

for filename in os.listdir(output):
    if filename.endswith(".png"):
        path = os.path.join(output, filename)
        img = face_recognition.load_image_file(path)
        locateFaceEncoding = face_recognition.face_encodings(img)
        
        if locateFaceEncoding and locateFaceEncoding[0].any():  # Check if at least one face is found
            knownFaces.append(locateFaceEncoding[0])
            knownFacesLabel.append(filename[:-4])  # Remove the file extension

openCapture = cv2.VideoCapture(0)
openCapture.set(3, 400)  # Width
openCapture.set(4, 400)  # Height

#skipF = 0  Adjust as needed
frameCount = 0

uniqueIdCount = len(knownFaces)
print(uniqueIdCount)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 400, 400)

while True:
    _, frame = openCapture.read()

    if frameCount:
        # Detect faces in the current frame
        locateFace = face_recognition.face_locations(frame)
        locateFaceEncoding = face_recognition.face_encodings(frame, locateFace)

        for faceLocation, faceEncoding in zip(locateFace, locateFaceEncoding):
            # Compare the current face encoding with the known face encodings
            matches = [face_recognition.compare_faces([currFace], faceEncoding)[0] for currFace in knownFaces]


            name = "Unknown"

            # If a match is found, use the label of the matched known face
            if True in matches:
                matchIndex = matches.index(True)
                name = knownFacesLabel[matchIndex]
            else:
                # Save the face as a known face
                uniqueIdCount += 1
                imgSavePath = os.path.join(output, f"face_{uniqueIdCount}.png")
                cv2.imwrite(imgSavePath, frame[faceLocation[0]:faceLocation[2], faceLocation[3]:faceLocation[1]])
                knownFaces.append(faceEncoding)
                knownFacesLabel.append(f"face_{uniqueIdCount}")
                name = f"face_{uniqueIdCount}"

            cv2.rectangle(frame, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (faceLocation[3] + 6, faceLocation[2] - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameCount += 1

openCapture.release()
cv2.destroyAllWindows()
