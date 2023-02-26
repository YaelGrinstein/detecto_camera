import cv2


# Load the image

# Define a function to handle mouse events
def select_roi(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        ref_point = [(x_start, y_start), (x_end, y_end)]
        if len(ref_point) == 2:
            roi = frame[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imshow("Cropped Image", roi)

def crop_image(frame):
    # Display the image
    cv2.imshow("Original Image", frame)
    # Set the callback function on mouse events
    cv2.namedWindow("Original Image")
    cv2.setMouseCallback("Original Image", select_roi)
    # Start the GUI loop
    while True:
        cv2.imshow("Original Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    # Close all windows
    cv2.destroyAllWindows()

