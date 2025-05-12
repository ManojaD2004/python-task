import cv2

# Open the default camera (usually webcam) → index 0
cap = cv2.VideoCapture("rtsp://test123:test123@192.168.1.8/stream1")

if not cap.isOpened():
    print("Cannot open camera")
    exit()


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Display the resulting frame
    # resize = ResizeWithAspectRatio(frame, width=1280)
    resized_frame = ResizeWithAspectRatio(frame, width=1280)
    cv2.imshow("Webcam Test", resized_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
