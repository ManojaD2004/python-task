import cv2
import queue
import threading

q = queue.Queue()


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


def Receive():
    print("start Receive")
    cap = cv2.VideoCapture("rtsp://test123:test123@192.168.1.8/stream1")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        q.put(frame)


if __name__ == "__main__":
    recv_thread = threading.Thread(target=Receive)
    recv_thread.daemon = True
    recv_thread.start()

    print("Start Displaying")
    while True:
        if not q.empty():
            frame = q.get()
            frame = ResizeWithAspectRatio(frame, width=1280)
            cv2.imshow("frame1", frame)

        # This must be in the main thread
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()