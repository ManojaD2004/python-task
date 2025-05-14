import cv2
import queue
import threading
import sys
from flask import Flask, Response, render_template_string
import os

q = queue.Queue()
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

camera_link = "rtsp://test123:test123@192.168.1.8/stream1"
print(camera_link)
cap = cv2.VideoCapture(camera_link, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def Receive():
    print("start Receive")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        q.put(frame)


def Display():
    while True:
        # print(q.qsize())
        if not q.empty():
            _ = q.get()

app = Flask(__name__)


def generate_frames():
    while True:
        if q.empty():
            continue
        frame = q.queue[0]
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


HTML_PAGE = """
<html>
  <head>
    <title>Live Camera</title>
  </head>
  <body>
    <img src="{{ url_for('video_feed') }}" width="800">
  </body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    recv_thread = threading.Thread(target=Receive)
    recv_thread.daemon = True
    recv_thread.start()
    display_thread = threading.Thread(target=Display)
    display_thread.daemon = True
    display_thread.start()

    app.run(host="0.0.0.0", port=5222, debug=False)
    # cv2.destroyAllWindows()
