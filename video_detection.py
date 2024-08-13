import cv2

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# We can use 0 for webcam, or provide a video file path Since no cam in my computer i used video file
cap = cv2.VideoCapture(r"C:\Users\Student\Desktop\Sutharsan_N_Internship\Face Recognition and Detection\img\v1.mp4")  # Change '0' to a file path to use a video file

#for output
#frame_width = 650
#frame_height = 650


while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

# frame = cv2.resize(frame, (frame_width, frame_height))
# Cropping rectangle
# crop_x, crop_y, crop_w, crop_h = 100, 100, 400, 300
# frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Drawing rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame with rectangles drawn around detected faces
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
