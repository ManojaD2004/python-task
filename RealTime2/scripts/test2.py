import os
import sys

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(
    r"D:\Full Stack Web Developer\Python Task\RealTime\Tensorflow\models\research"
)

import queue
import time
import threading
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
import tensorflow as tf

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
# print(cv2.getBuildInformation())
cap = cv2.VideoCapture("rtsp://test123:test123@192.168.1.8/stream1")
# cap = cv2.VideoCapture(0)
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

import cv2

i1 = 1

def draw_boxes_opencv(image, boxes, classes, scores, category_index, threshold=0.5):
    height, width, _ = image.shape
    # print(len(boxes))
    for i in range(min(len(boxes), 1000)):  # draw up to 20 boxes or adjust as needed
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
            label = f"{class_name}: {int(score * 100)}%"

            # If box coords are normalized → denormalize
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw rectangle
            print(f"Score: {score}")
            print(f"Coords (normalized): {box}")
            print(
                f"Coords (pixels): xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"
            )
            print(f"Frame size: {width}x{height}")
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Put label above box
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
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
                0.6,
                (0, 0, 0),
                2,
            )
            # global i1
            # cv2.imwrite("test_frame"+str(i1)+".jpg", image)
            # i1 += 1
    return image


def Receive():
    print("start Reveive")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def Display():
    print("Start Displaying")
    while True:
        if q.empty() != True:
            frame = q.get()
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
            num_detections = int(detections.pop("num_detections"))
            detections = {
                key: value[0, :num_detections].numpy() for key, value in detections.items()
            }
            detections["num_detections"] = num_detections

            # detection_classes should be ints.
            detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # viz_utils.visualize_boxes_and_labels_on_image_array(
            #     image_np_with_detections,
            #     detections["detection_boxes"],
            #     detections["detection_classes"] + label_id_offset,
            #     detections["detection_scores"],
            #     category_index,
            #     use_normalized_coordinates=True,
            #     max_boxes_to_draw=5,
            #     min_score_thresh=0.1,
            #     agnostic_mode=False,
            # )
            image_np_with_detections = draw_boxes_opencv(
                image_np_with_detections,
                detections["detection_boxes"],
                detections["detection_classes"] + label_id_offset,
                detections["detection_scores"],
                category_index,
                threshold=0.5,  # or any min_score_thresh
            )
            # cv2.imwrite("test_frame.jpg", image_np_with_detections)
            # cv2.imwrite("test_frame1.jpg", frame)
            # print("in")
            # resized_frame = ResizeWithAspectRatio(image_np_with_detections, width=1280)
            cv2.imshow("Object Detection", cv2.resize(image_np_with_detections, (800, 600)))
            # cv2.imshow(
            #     "Object Detection", cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
            # )
            # cv2.imshow("Object Detection", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main1():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections

        # detection_classes should be ints.
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        # viz_utils.visualize_boxes_and_labels_on_image_array(
        #     image_np_with_detections,
        #     detections["detection_boxes"],
        #     detections["detection_classes"] + label_id_offset,
        #     detections["detection_scores"],
        #     category_index,
        #     use_normalized_coordinates=True,
        #     max_boxes_to_draw=5,
        #     min_score_thresh=0.1,
        #     agnostic_mode=False,
        # )
        # print("max score:", max(detections["detection_scores"]))

        image_np_with_detections = draw_boxes_opencv(
            image_np_with_detections,
            detections["detection_boxes"],
            detections["detection_classes"] + label_id_offset,
            detections["detection_scores"],
            category_index,
            threshold=0.5,  # or any min_score_thresh
        )
        # if first_time:
        #     # print(detections)
        #     print(category_index)
        #     print(np.all(image_np == image_np_with_detections))
        #     print(detections["detection_scores"])

        # first_time = False
        # cv2.imwrite("test_frame.jpg", image_np_with_detections)
        # cv2.imwrite("test_frame1.jpg", frame)
        # print("in")
        # resized_frame = ResizeWithAspectRatio(image_np_with_detections, width=1280)
        cv2.imshow("Object Detection", cv2.resize(image_np_with_detections, (800, 600)))
        # cv2.imshow(
        #     "Object Detection", cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
        # )
        # cv2.imshow("Object Detection", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            break


if __name__ == "__main__":
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
