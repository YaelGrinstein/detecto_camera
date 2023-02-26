import cv2
import numpy as np

# Set the number of inner corners on the checkerboard pattern
checkerboard_size = (8, 6)

# Set the size of the squares on the checkerboard pattern in millimeters
square_size = 24

# Set the number of calibration images to use
num_images = 20

# Create a list to store the 3D points of the checkerboard pattern
objpoints = []

# Create a list to store the 2D points of the checkerboard pattern in each image
imgpoints = []

# Generate the 3D coordinates of the checkerboard corners
objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# Capture calibration images from the Foscam camera
cap = cv2.VideoCapture('rtsp://pi:pi123456!@199.203.102.124:86/videoSub')
for i in range(num_images):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow('Calibration Image', img)
        cv2.waitKey(500)

# Close the camera and destroy the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Perform camera calibration
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the calibration results
print('Intrinsic matrix (K):\n', K)
print('Distortion coefficients (D):\n', D)

# Save the calibration results to a file
np.savetxt('K.txt', K)
np.savetxt('D.txt', D)
