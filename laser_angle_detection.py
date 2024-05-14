import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def angle_between_points(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

def find_mean_direction(directions):
    mean_direction = np.arctan2(np.mean(np.sin(directions)), np.mean(np.cos(directions)))
    return mean_direction

def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)

def are_directions_similar(angle1, angle2, threshold_degrees):
    # Calculate the absolute difference between the angles
    angle_diff = np.abs(angle1 - angle2)
    # Normalize the angle difference to be in the range [0, pi]
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    # Convert the threshold from degrees to radians
    threshold_radians = np.radians(threshold_degrees)
    # Check if the angle difference is within the threshold
    return angle_diff <= threshold_radians

def combine_lines(lines, directions, threshold_degrees):
    mean_direction = find_mean_direction(directions)

    # Find the mean starting and ending points of all lines
    start_x = np.mean([line[0][0] for line in lines])
    start_y = np.mean([line[0][1] for line in lines])
    end_x = np.mean([line[0][2] for line in lines])
    end_y = np.mean([line[0][3] for line in lines])

    # Calculate the length of the longest line
    max_length_line = max(lines, key=lambda x: abs(x[0][1] - x[0][3]))
    max_length = abs(max_length_line[0][1] - max_length_line[0][3])

    # Extend the combined line from the midpoint in the mean direction
    combined_line = np.array([[start_x, start_y, end_x, end_y]])
    dx = max_length * np.cos(mean_direction)
    dy = max_length * np.sin(mean_direction)
    combined_line[0][2] = combined_line[0][0] + dx
    combined_line[0][3] = combined_line[0][1] + dy

    return combined_line

def combine_similar_lines(lines_coordinates, directions, threshold_degrees):
    # Convert lines_coordinates to numpy array
    lines = [np.array(line) for line in lines_coordinates]

    # Identify lines with similar directions
    similar_lines = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            if are_directions_similar(directions[i], directions[j], threshold_degrees):
                if i not in similar_lines:
                    similar_lines.append(i)
                if j not in similar_lines:
                    similar_lines.append(j)

    # Combine lines with similar directions into a single mean line
    if len(similar_lines) > 1:
        combined_line = combine_lines([lines[i] for i in similar_lines], [directions[i] for i in similar_lines], threshold_degrees)
        for idx in similar_lines:
            lines[idx] = combined_line
    # If there are no similar lines, leave them unchanged

    return lines

def line_length(line):
    return np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2)

def angle_between_lines(line1, line2):
    dx1 = line1[0][2] - line1[0][0]
    dy1 = line1[0][3] - line1[0][1]
    dx2 = line2[0][2] - line2[0][0]
    dy2 = line2[0][3] - line2[0][1]
    angle = np.arccos((dx1*dx2 + dy1*dy2) / (np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2)))
    return np.degrees(angle)

def detect(img):
    #glare removal
    overlay = img.copy()
    hh, ww = img.shape[:2]
    lower = (220,220,220)
    upper = (255,255,255)
    thresh = cv2.inRange(img, lower, upper)
    black = np.zeros([hh + 2, ww + 2], np.uint8)
    mask = thresh.copy()
    mask_b = cv2.floodFill(mask, black, (0,0), 0, 0, 0, flags=8)[1]
    indices = np.where(mask_b==255)
    img[indices[0], indices[1], :] = [255, 255, 127]
    alpha = 0.7
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    #seperating red channel from RGB
    B,G,R = cv2.split(img_new)
    #bluring using average filter
    average = cv2.blur(R,(1,3))
    #converting into binary image
    masked = cv2.inRange(average, 220, 255)
    #creating a custom kernel to dilate in diagonal
    kernel_diag = np.zeros((3, 5), np.uint8)
    n = 1
    np.fill_diagonal(kernel_diag, n)
    np.fill_diagonal(np.fliplr(kernel_diag), n)
    dilate_diag = cv2.dilate(masked,kernel_diag,iterations = 5)
    #dilate in cross
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,5))
    dilate_cross = cv2.dilate(dilate_diag ,kernel,iterations = 20)
    #erode in cross
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,5))
    erode_cross = cv2.erode(dilate_cross,kernel,iterations = 25)
    #edge detection
    edges = cv2.Canny(erode_cross, 50, 200, None, 3)
    #line detection
    threshold_degrees = 0.2
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=10, maxLineGap=250)
    directions = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            direction = angle_between_points(l[0], l[1],l[2], l[3])
            directions.append(direction)
        lines_combined = combine_similar_lines(linesP, directions, threshold_degrees)
        lines_tuples = [tuple(line.flatten()) for line in lines_combined]
        # Create a set to remove duplicates
        lines_set = set(lines_tuples)
        # Convert the set back to a list of NumPy arrays
        unique_lines = [np.array(line).reshape(-1, 4) for line in lines_set]
        #covert float to int
        n_lines = [arr.astype(int) for arr in unique_lines]
        if n_lines is not None:
            for i in range(0, len(n_lines)):
                print(n_lines[i])
                l = n_lines[i][0]
                cv2.line(img , (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        # Calculate lengths of all lines
        line_lengths = [line_length(line) for line in n_lines]
        # Sort lines by length
        sorted_lines = [line for _, line in sorted(zip(line_lengths, n_lines), reverse=True)]
        if len(sorted_lines) <= 2:
            # Select the two longest lines
            longest_lines = sorted_lines[:2]
            angle_degrees = None  # Initialize angle_degrees
            if len(longest_lines) >= 2:
                angle_degrees = angle_between_lines(longest_lines[0], longest_lines[1])
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
    angle = detect(frame)

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
