import cv2
import numpy as np

# Function to find the angle of the sheet metal
def find_angle(image):
    # Convert the image to grayscale   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Calculate the angle of the sheet metal
    angle = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_radians = np.arctan2(y2 - y1, x2 - x1)
            angle_degrees = np.degrees(angle_radians)
            # Ensure angle is between 0 and 180 degrees
            if angle_degrees < 0:
                angle_degrees += 180
            # Convert angle to be in range [0, 180)
            angle = angle_degrees if angle_degrees < 180 else 0
            # Mirror the angle for left side
            if angle > 0:
                angle = 180 - angle  
            # Draw the detected lines
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return angle
    else:
        return None

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Flip the camera image
    frame = cv2.flip(frame, -1)  # -1 for upside down, 1 for mirror the image

    # Call the function to find the angle of the sheet metal
    angle = find_angle(frame)

    # Save the frame to a specified path
    if angle is not None:
        print("Angle: {:.2f} degrees".format(angle))
        cv2.putText(frame, f'Angle: {angle:.2f} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 240, 0), 2)
        cv2.imwrite('image.jpg', frame)
    else:
        print("Not enough lines found to measure angle.")
    break

# Release the capture
cap.release()
cv2.destroyAllWindows()
  