import cv2
import numpy as np
import time

# Function to find the angle of the sheet metal
def find_angle(image):
    # Convert the image to grayscale   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Calculate the angle of the sheet metal
    angle = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            # Ensure angle is between 0 and 180 degrees
            if angle < 0:
                angle += 180
            # Draw the detected lines
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return angle
    else:
        return None

# Capture video from the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
    fps = int(cap.get(5))
    print("fps:", fps)

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    
    # Resize the frame
    frame = cv2.resize(frame, (720, 480))

    #fliping the camera
    frame = cv2.flip(frame, -1) #-1 for upside down, 1 for mirror the image

    # Call the function to find the angle of the sheet metal
    angle = find_angle(frame)

    # Display the angle value on the camera image
    if angle is not None:
       cv2.putText(frame, f'Angle: {angle:.2f} degrees', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 0), 1)

    # Display the frame
    cv2.imshow('Angle measurement Camera', frame)

    # Press 'esc' to quit the program
    if cv2.waitKey(1) & 0xFF == 27:
        # Display 'closing the window' text on the camera image
        cv2.putText(frame, 'Closing the window...', (90, 160), cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 1)
        cv2.imshow('Angle measurement Camera', frame)
        cv2.waitKey(2000)  # Delay for 2 seconds
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
