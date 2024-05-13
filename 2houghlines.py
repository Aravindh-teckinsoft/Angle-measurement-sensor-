import cv2
import numpy as np
import math

# Function to detect and find angle between two lines using Hough Line Transform
def detect_and_measure_angle(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply erosion to the edges
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=5)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Initialize variables to store line parameters
    rho1, theta1, rho2, theta2 = 0, 0, 0, 0

    # Draw detected lines and find parameters for two longest lines
    if lines is not None:
        lines = lines.squeeze(axis=1)  # Remove unnecessary dimension
        if len(lines) >= 2:
            longest_lines = sorted(lines, key=lambda x: x[0], reverse=True)[:2]
            for rho, theta in longest_lines:
                a = np.cos(theta)
                b = np.sin(theta)
                # Calculate start and end points of the line
                x0 = a * rho
                y0 = b * rho
                length = 1000
                x1 = int(x0 + length * (-b))
                y1 = int(y0 + length * (a))
                x2 = int(x0 - length * (-b))
                y2 = int(y0 - length * (a))
                # Draw the line on the image
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Store parameters of the two longest lines
                if rho == longest_lines[0][0]:
                    rho1, theta1 = rho, theta
                else:
                    rho2, theta2 = rho, theta

    # Calculate angle between the two lines
    if rho1 != 0 and rho2 != 0:
        angle_radians = abs(theta2 - theta1)
        if angle_radians > np.pi / 2:
            angle_radians = np.pi - angle_radians
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees
    else:
        return None

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 540))
    frame = cv2.flip(frame, -1)  # 1 for horizontal flip
    # Call the function to detect and measure angle between two lines
    angle = detect_and_measure_angle(frame)

    # Display the angle value on the camera image
    if angle is not None:
        cv2.putText(frame, f'Angle: {angle:.2f} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 240, 0), 2)

    # Display the frame
    cv2.imshow('Angle measurement Camera', frame)

    # Press 'esc' to quit the program
    if cv2.waitKey(1) & 0xFF == 27:
        # Display 'closing the window' text on the camera image
        cv2.putText(frame, 'Closing the window...', (150, 240), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Angle measurement Camera', frame)
        cv2.waitKey(2000)  # Delay for 2 seconds
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
