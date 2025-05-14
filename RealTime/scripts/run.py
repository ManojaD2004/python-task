import os
import sys
import json

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(r"./Tensorflow/models/research")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

import queue
import time
import threading
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template_string
from simple_facerec import SimpleFacerec
from confluent_kafka import Producer

WORKSPACE_PATH = "Tensorflow/workspace"
SCRIPTS_PATH = "Tensorflow/scripts"
APIMODEL_PATH = "Tensorflow/models"
ANNOTATION_PATH = WORKSPACE_PATH + "/annotations"
IMAGE_PATH = WORKSPACE_PATH + "/images"
MODEL_PATH = WORKSPACE_PATH + "/models"
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + "/pre-trained-models"
CONFIG_PATH = MODEL_PATH + "/my_ssd_mobnet/pipeline.config"
CHECKPOINT_PATH = MODEL_PATH + "/my_ssd_mobnet/"

CUSTOM_MODEL_NAME = "my_ssd_mobnet"

CONFIG_PATH = MODEL_PATH + "/" + CUSTOM_MODEL_NAME + "/pipeline.config"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs["model"], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, "ckpt-6")).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(
    ANNOTATION_PATH + "/label_map.pbtxt"
)


# Setup capture
stream_url = sys.argv[2]
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

width1 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


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


q = queue.Queue()
detection_lock = threading.Lock()
label_id_offset = 1
i1 = 1
detections1 = None
last_label = "NoFolding"
total_task = 0
productive_time = 0
total_time = 0
room_id = sys.argv[3]
camera_id = sys.argv[4]
interval_sec = float(sys.argv[5])
c = {}
d = {}
producer = None
latest_face_locations = []
latest_face_names = []
prev_face_data = {"faceDetected": False}
with open(sys.argv[1], "r") as file1:
    data1 = json.load(file1)
    c = data1
    
for i in c.keys():
    d[i] = {"prodTime": 0, "totalTime": 0, "taskDone": 0}

sfr = SimpleFacerec()
sfr.load_encoding_images(c)

# Flask
app = Flask(__name__)


def generate_frames():
    while True:
        if q.empty():
            continue
        frame = q.queue[0]
        image_np = np.array(frame)
        image_np_with_detections = image_np.copy()
        face_locations = latest_face_locations.copy()
        face_names = latest_face_names.copy()
        if detections1:
            # print("in 2")
            image_np_with_detections = draw_boxes_opencv(
                image_np_with_detections,
                detections1["detection_boxes"],
                detections1["detection_classes"] + label_id_offset,
                detections1["detection_scores"],
                category_index,
                # face_locations,
                # face_names,
                threshold=0.75,  # or any min_score_thresh
            )
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc
            if name == "Unknown":
                cv2.rectangle(image_np_with_detections, (x1, y1), (x2, y2), (0, 0, 200), 2)
                cv2.putText(
                    image_np_with_detections,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 0, 200),
                    2,
                )
                continue
            textToDisplay = c[name]["empName"]
            cv2.rectangle(image_np_with_detections, (x1, y1), (x2, y2), (0, 0, 200), 2)
            cv2.putText(
                image_np_with_detections,
                textToDisplay,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 200),
                2,
            )

        head_count = len(face_locations)
        cv2.putText(
            image_np_with_detections,
            f"Head Count: {head_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        ret, buffer = cv2.imencode(".jpg", image_np_with_detections)
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

# Detect Fn
def update_metadata(classes, scores, category_index, threshold=0.5, draw_limit=2):
    # print(len(boxes))
    global last_label
    global total_time
    global total_task
    global productive_time
    face_locations = latest_face_locations.copy()
    face_names = latest_face_names.copy()
    for i in range(
        min(len(classes), draw_limit)
    ):  # draw up to 20 boxes or adjust as needed
        score = scores[i]
        # print(score, i)
        if score >= threshold:
            # print("in", score)
            class_id = int(classes[i])
            total_time += 0.5
            # print("Class id: ", class_id)
            if category_index[class_id]["name"] == "Folding":
                productive_time += 0.5
            if (
                last_label == "NoFolding"
                and category_index[class_id]["name"] == "Folding"
            ):
                total_task += 1
            last_label = category_index[class_id]["name"]


def draw_boxes_opencv(
    image, boxes, classes, scores, category_index, threshold=0.5, draw_limit=2
):
    height, width, _ = image.shape
    # print(len(boxes))
    for i in range(
        min(len(boxes), draw_limit)
    ):  # draw up to 20 boxes or adjust as needed
        score = scores[i]
        # print(score, i)
        if score >= threshold:
            # print("in", score)
            box = boxes[i]
            class_id = int(classes[i])
            class_name = (
                category_index[class_id]["name"]
                if class_id in category_index
                else str(class_id)
            )
            label = f"{class_name}: {int(score * 100)}%, Total Time: {total_time}, Prod Time: {productive_time}, Task Done: {total_task}"

            # If box coords are normalized → denormalize
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw rectangle
            # print(f"Score: {score}")
            # print(f"Coords (normalized): {box}")
            # print(
            #     f"Coords (pixels): xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"
            # )
            # print(f"Frame size: {width}x{height}")
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Put label above box
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(
                image,
                (xmin, label_ymin - label_size[1] - 10),
                (xmin + label_size[0], label_ymin),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (xmin, label_ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
            # global i1
            # cv2.imwrite("test_frame"+str(i1)+".jpg", image)
            # i1 += 1
    return image


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

def json_log():
    global prev_face_data
    face_data = {
        "faceDetected": len(latest_face_locations) > 0,
        "timestamp": time.time(),
        "headCount": len(latest_face_locations),
        "empIds": latest_face_names,
        "roomId": room_id,
        "cameraId": camera_id,
        "totalTime": total_time,
        "prodTime": productive_time,
        "taskComplete": total_task,
        "taskName": "Packing"
    }
    print(face_data)
    # if (face_data["faceDetected"] or prev_face_data["faceDetected"]) and producer:
    #     producer.produce(
    #         "camera-job", key=str(camera_id), value=json.dumps(face_data), callback=kafka_delivery_report
    #     )
    #     producer.flush()
    prev_face_data = face_data

def do_detect():
    global detections1
    global latest_face_locations
    global latest_face_names
    global room_id
    global camera_id
    while True:
        if not q.empty():
            frame = q.queue[0]
            # print("in 1 start")
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32
            )
            # print("in 1 1 start")
            # start_time = time.time()
            detections = detect_fn(input_tensor)
            # print("in 1 1 end", time.time() - start_time, " sec")
            num_detections = int(detections.pop("num_detections"))
            detections = {
                key: value[0, :num_detections].numpy()
                for key, value in detections.items()
            }
            detections["num_detections"] = num_detections
            # detection_classes should be ints.
            detections["detection_classes"] = detections["detection_classes"].astype(
                np.int64
            )
            # print("in 1 end")
            face_locations, face_names = sfr.detect_known_faces(frame.copy())
            with detection_lock:
                latest_face_locations = face_locations
                latest_face_names = face_names
                detections1 = detections
                update_metadata(
                    detections1["detection_classes"] + label_id_offset,
                    detections1["detection_scores"],
                    category_index,
                    threshold=0.75,
                )
            # JSON log
            json_log()
            time.sleep(interval_sec)

def kafka_delivery_report(err, msg):
    if err is not None:
        print("Delivery failed:", err)
    else:
        print("Message delivered to", msg.topic(), msg.partition())


def main1():
    print(sys.argv)
    kafka_broker_url = sys.argv[6]
    # print(sys.argv, kafka_broker_url)
    print(kafka_broker_url)
    # conf = {"bootstrap.servers": kafka_broker_url}
    # global producer
    # producer = Producer(conf)
    start_time = time.time()
    print("dummy detect started")
    dummy_input = tf.convert_to_tensor(
        np.zeros((1, height1, width1, 3), dtype=np.float32)
    )
    _ = detect_fn(dummy_input)
    print("dummy detect ended", time.time() - start_time, " sec")
    recv_thread = threading.Thread(target=Receive)
    recv_thread.daemon = True
    recv_thread.start()
    display_thread = threading.Thread(target=Display)
    display_thread.daemon = True
    display_thread.start()
    detect_thread = threading.Thread(target=do_detect)
    detect_thread.daemon = True
    detect_thread.start()
    app.run(host="0.0.0.0", port=5222, debug=False)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main1()
