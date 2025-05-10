import cv2
import queue
import time
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
    print("start Reveive")
    cap = cv2.VideoCapture("rtsp://test123:test123@192.168.1.8/stream1")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Display():
    print("Start Displaying")
    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow("frame1", ResizeWithAspectRatio(frame, width=1280))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
