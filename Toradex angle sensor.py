import cv2
import numpy as np


# Function to find the angle between two lines
def find_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    angle1 = np.arctan2(y2 - y1, x2 - x1)
    angle2 = np.arctan2(y4 - y3, x4 - x3)
    angle_deg = np.degrees(angle2 - angle1)
    if angle_deg > 180:
        angle_deg -= 180
    return abs(angle_deg)


# Function to detect lines in the frame
def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return lines


# Capture image from the camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Find lines in the image
lines = detect_lines(frame)

# Find the two longest lines
if lines is not None and len(lines) >= 2:
    lines = sorted(lines, key=lambda x: np.linalg.norm(x[0][2:] - x[0][:2]), reverse=True)[:2]
    line1 = lines[0][0]
    line2 = lines[1][0]

    # Calculate angle between the two lines
    angle = find_angle(line1, line2)

    # fliping the camera
    frame = cv2.flip(frame, -1)  # -1 for upside down, 1 for mirror the image

    # Print the angle
    print("Angle of the sheet metal: {:.2f} degrees".format(angle))

    # Draw the lines on the image
    cv2.line(frame, (line1[0], line1[1]), (line1[2], line1[3]), (0, 255, 0), 2)
    cv2.line(frame, (line2[0], line2[1]), (line2[2], line2[3]), (0, 255, 0), 2)
    # Display the angle on the image
    cv2.putText(frame, f'Angle: {angle:.2f} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the image with angle overlaid
    cv2.imwrite('sheet_metal_with_angle.jpg', frame)

else:
    print("Not enough lines found to measure angle.")

# Release the capture
cap.release()
cv2.destroyAllWindows()
