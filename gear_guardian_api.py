from flask import Flask, request, jsonify, send_file
import cv2
import json
import numpy as np

import os
import tensorflow as tf
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes, draw_box_with_id
from object_tracking.application_util import preprocessing, generate_detections as gdet
from object_tracking.deep_sort import nn_matching
from object_tracking.deep_sort.detection import Detection
from object_tracking.deep_sort.tracker import Tracker

from tensorflow.keras.models import load_model
from tqdm import tqdm
import numpy as np
from object_tracking.application_util import preprocessing
from object_tracking.deep_sort import nn_matching
from object_tracking.deep_sort.detection import Detection
from object_tracking.deep_sort.tracker import Tracker
from object_tracking.application_util import generate_detections as gdet
from utils.bbox import draw_box_with_id

import warnings

warnings.filterwarnings("ignore")

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from multiprocessing import Queue
from app import App

app = Flask(__name__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

config = {
  "model": {
    "min_input_size": 288,
    "max_input_size": 448,
    "anchors": [
      33,
      34,
      52,
      218,
      55,
      67,
      92,
      306,
      96,
      88,
      118,
      158,
      153,
      347,
      209,
      182,
      266,
      359
    ],
    "labels": ["capacete", "pessoa com capacete", "pessoa sem capacete"]
  },

  "train": {
    "train_image_folder": "train_image_folder/",
    "train_annot_folder": "train_annot_folder/",
    "cache_name": "helmet_train.pkl",

    "train_times": 8,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "nb_epochs": 100,
    "warmup_epochs": 3,
    "ignore_thresh": 0.5,
    "gpus": "0,1",

    "grid_scales": [1, 1, 1],
    "obj_scale": 5,
    "noobj_scale": 1,
    "xywh_scale": 1,

    "tensorboard_dir": "logs",
    "saved_weights_name": "full_yolo3_helmet_and_person.h5",
    "debug": True
  },

  "valid": {
    "valid_image_folder": "",
    "valid_annot_folder": "",
    "cache_name": "",

    "valid_times": 1
  }
}

def draw_box_with_id(image, bbox, track_id, label, labels):
    color = None
    if label == 1:  # with helmet
        color = (0, 255, 0)  # green
    elif label == 2:  # without helmet
        color = (0, 0, 255)  # red
    else:  # for other categories if there are any
        color = (255, 255, 0)  # some other color, e.g., blue

    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    cv2.putText(image, labels[label], (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_video(video_path, output_video_path, object_label):
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45

    infer_model = load_model(config['train']['saved_weights_name'])
    encoder = gdet.create_box_encoder('mars-small128.pb', batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3, None)
    tracker = Tracker(metric)

    video_reader = cv2.VideoCapture(video_path)

    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'MP4V'),  # Or use 'XVID' for .avi format
        video_reader.get(cv2.CAP_PROP_FPS),
        (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    object_found = False
    labels = config['model']['labels']

    while True:
        ret_val, image = video_reader.read()
        if not ret_val:
            break

        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
        boxs = [[box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin] for box in boxes]
        features = encoder(image, boxs)
        detections = [Detection(b, box.c, f, box.label) for b, box, f in zip(boxs, boxes, features)]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            draw_box_with_id(image, bbox, track.track_id, track.label, labels)
            
            # Check if the desired object was found in the video frame
            if labels[track.label] == object_label:
                object_found = True

        video_writer.write(image)

    video_reader.release()
    video_writer.release()

    return object_found

@app.route('/predict/video', methods=['POST'])
def predict_video():
    object_label = 'helmet'  # The desired object you are looking for, you can modify this

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_video_file(file.filename):
        video_bytes = file.read()
        temp_video_name = "temp_video.mp4"
        output_video_name = "output_video.mp4"

        with open(temp_video_name, "wb") as f:
            f.write(video_bytes)

        object_found = process_video(temp_video_name, output_video_name, object_label)

        os.remove(temp_video_name)

        # Send both the processed video and the JSON response
        # response = send_file(output_video_name, mimetype='video/mp4', as_attachment=True)
        # response.headers['Object-Found'] = str(object_found)  # Using a header to convey the result
        
        response = {"object_found": object_found}
        return response

    return jsonify({"error": "Unsupported file format"}), 415


def process_image(image_path, object_label):
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45

    infer_model = load_model(config['train']['saved_weights_name'])
    encoder = gdet.create_box_encoder('mars-small128.pb', batch_size=1)

    image = cv2.imread(image_path)

    boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

    boxs = [[box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin] for box in boxes]
    features = encoder(image, boxs)
    detections = [Detection(b, box.c, f, box.label) for b, box, f in zip(boxs, boxes, features)]

    labels = config['model']['labels']
    object_found = False

    for detection in detections:
        if labels[detection.label] == object_label:
            object_found = True

        bbox = detection.to_tlbr()
        draw_box_with_id(image, bbox, None, detection.label, labels)  # Track id is None for single image processing

    output_image_path = "output_" + image_path
    cv2.imwrite(output_image_path, image)

    return output_image_path, object_found


@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    object_label = request.form.get('object_label', 'helmet')  # Default to 'helmet' if no object_label provided

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_image_file(file.filename):  # You need a helper function (like allowed_video_file) for images
        image_bytes = file.read()
        temp_image_name = "temp_image.jpg"  # Or use appropriate extension

        with open(temp_image_name, "wb") as f:
            f.write(image_bytes)

        output_image_path, object_found = process_image(temp_image_name, object_label)

        os.remove(temp_image_name)

        # Add a custom header to indicate if the object was found
        response = send_file(output_image_path, mimetype='image/jpeg', as_attachment=False)
        response.headers["Object-Found"] = str(object_found)
        # response = {"object_found": object_found}

        # Clean up by deleting the output image after sending it to the client
        os.remove(output_image_path)

        return response

    return jsonify({"error": "Unsupported file format"}), 415


def allowed_image_file(filename):
    # Modify this to fit your required extensions
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    # Modify this to fit your required extensions
    ALLOWED_VIDEO_EXTENSIONS = set(['mp4', 'avi', 'mov'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
