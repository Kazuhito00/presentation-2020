#!/usr/bin/env python
# -*- coding: utf-8 -*-

import eel
import cv2 as cv
import csv
import base64
import copy

import tensorflow as tf
import numpy as np

from utils import CvDrawText


@eel.expose
def slide_change_event(val):
    global current_slide
    current_slide = val


def run_inference_single_image(image, inference_func):
    tensor = tf.convert_to_tensor(image)
    output = inference_func(tensor)

    output['num_detections'] = int(output['num_detections'][0])
    output['detection_classes'] = output['detection_classes'][0].numpy()
    output['detection_boxes'] = output['detection_boxes'][0].numpy()
    output['detection_scores'] = output['detection_scores'][0].numpy()
    return output


def demo01(inference_func, frame):
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    debug_image = copy.deepcopy(frame)

    frame = frame[:, :, [2, 1, 0]]  # BGR2RGB
    image_np_expanded = np.expand_dims(frame, axis=0)

    output = run_inference_single_image(image_np_expanded, inference_func)

    num_detections = output['num_detections']
    for i in range(num_detections):
        score = output['detection_scores'][i]
        bbox = output['detection_boxes'][i]
        class_id = output['detection_classes'][i].astype(np.int)

        if score < 0.75:
            continue

        # 検出結果可視化 ###################################################
        x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
        x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

        # バウンディングボックス(長い辺にあわせて正方形を表示)
        x_len = x2 - x1
        y_len = y2 - y1
        square_len = x_len if x_len >= y_len else y_len
        square_x1 = int(((x1 + x2) / 2) - (square_len / 2))
        square_y1 = int(((y1 + y2) / 2) - (square_len / 2))
        square_x2 = square_x1 + square_len
        square_y2 = square_y1 + square_len

        cv.rectangle(debug_image, (square_x1, square_y1),
                     (square_x2, square_y2), (0, 255, 0), 2)

        font_size = square_len
        class_string = ''
        if (class_id - 1) == 0:
            class_string = 'L'
        elif (class_id - 1) == 1:
            class_string = 'R'

        font_path = './utils/font/x12y20pxScanLine.ttf'
        debug_image = CvDrawText.puttext(
            debug_image,
            class_string,
            (square_x1 + int(font_size / 10), square_y1 + int(font_size / 10)),
            font_path,
            font_size,
            (0, 255, 0),
        )
    return debug_image


def demo02(inference_func, frame):
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    debug_image = copy.deepcopy(frame)

    frame = frame[:, :, [2, 1, 0]]  # BGR2RGB
    image_np_expanded = np.expand_dims(frame, axis=0)

    output = run_inference_single_image(image_np_expanded, inference_func)

    num_detections = output['num_detections']
    for i in range(num_detections):
        score = output['detection_scores'][i]
        bbox = output['detection_boxes'][i]
        # class_id = output['detection_classes'][i].astype(np.int)

        if score < 0.75:
            continue

        # 検出結果可視化 ###################################################
        x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
        x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

        cv.putText(debug_image, '{:.3f}'.format(score), (x1 - 1, y1 - 16),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
        cv.putText(debug_image, '{:.3f}'.format(score), (x1 + 1, y1 - 14),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
        cv.putText(debug_image, '{:.3f}'.format(score), (x1, y1 - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (139, 0, 0), 2, cv.LINE_AA)
        cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 255, 255), 4)
        cv.rectangle(debug_image, (x1, y1), (x2, y2), (139, 0, 0), 2)
    return debug_image


def demo03(inference_func, frame, labels):
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    debug_image = copy.deepcopy(frame)

    frame = frame[:, :, [2, 1, 0]]  # BGR2RGB
    image_np_expanded = np.expand_dims(frame, axis=0)

    output = run_inference_single_image(image_np_expanded, inference_func)

    num_detections = output['num_detections']
    for i in range(num_detections):
        score = output['detection_scores'][i]
        bbox = output['detection_boxes'][i]
        class_id = output['detection_classes'][i].astype(np.int)

        # 検出閾値未満のバウンディングボックスは捨てる
        if score < 0.6:
            continue

        x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
        x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

        # バウンディングボックス(長い辺にあわせて正方形を表示)
        x_len = x2 - x1
        y_len = y2 - y1
        square_len = x_len if x_len >= y_len else y_len
        square_x1 = int(((x1 + x2) / 2) - (square_len / 2))
        square_y1 = int(((y1 + y2) / 2) - (square_len / 2))
        square_x2 = square_x1 + square_len
        square_y2 = square_y1 + square_len
        cv.rectangle(debug_image, (square_x1, square_y1),
                     (square_x2, square_y2), (255, 255, 255), 4)
        cv.rectangle(debug_image, (square_x1, square_y1),
                     (square_x2, square_y2), (0, 0, 0), 2)

        # 印の種類
        font_path = './utils/font/衡山毛筆フォント.ttf'
        font_size = int(square_len / 2)
        debug_image = CvDrawText.puttext(
            debug_image, labels[class_id][1],
            (square_x2 - font_size, square_y2 - font_size), font_path,
            font_size, (185, 0, 0))

        # 検出スコア(表示オプション有効時)
        if False:
            font_size = int(square_len / 8)
            debug_image = CvDrawText.puttext(debug_image,
                                             '{:.3f}'.format(score),
                                             (square_x1 + int(font_size / 4),
                                              square_y1 + int(font_size / 4)),
                                             font_path, font_size, (185, 0, 0))
    return debug_image


# メイン処理 #############################################################
# WebSlides側の頁数保持用変数 #############################################
global current_slide
current_slide = 1

# カメラ起動 #############################################################
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

ret, dummy_image = cap.read()
if not ret:
    exit()

# モデルロード ###########################################################
DEFAULT_FUNCTION_KEY = 'serving_default'

# 01：手検出
loaded_model01 = tf.saved_model.load('model/01_HandDetection/saved_model')
inference_func01 = loaded_model01.signatures[DEFAULT_FUNCTION_KEY]
demo01(inference_func01, dummy_image)

# 02：FingerFrame検出
loaded_model02 = tf.saved_model.load('model/02_FingerFrame/saved_model')
inference_func02 = loaded_model02.signatures[DEFAULT_FUNCTION_KEY]
demo02(inference_func02, dummy_image)

# 03：NARUTO印検出
loaded_model03 = tf.saved_model.load('model/03_NarutoHandSign/saved_model')
inference_func03 = loaded_model03.signatures[DEFAULT_FUNCTION_KEY]
# ラベル読み込み
with open('model/03_NarutoHandSign/labels.csv', encoding='utf8') as f:
    labels = csv.reader(f)
    labels = [row for row in labels]
demo03(inference_func03, dummy_image, labels)

# Eel起動 ###############################################################
eel.init('web')
eel.start(
    'index.html',
    mode='chrome',
    # cmdline_args=['--start-fullscreen', '--browser-startup-dialog'])
    cmdline_args=['--start-fullscreen'],
    block=False)

while True:
    eel.sleep(0.01)

    # カメラキャプチャ ###################################################
    ret, frame = cap.read()
    if not ret:
        continue

    if 7 <= current_slide <= 8:
        frame = demo01(inference_func01, frame)
    if 9 <= current_slide <= 10:
        frame = demo02(inference_func02, frame)
    if 11 <= current_slide <= 14:
        frame = demo03(inference_func03, frame, labels)

    # UI側へ転送
    _, imencode_image = cv.imencode('.jpg', frame)
    base64_image = base64.b64encode(imencode_image)
    eel.set_base64image("data:image/jpg;base64," +
                        base64_image.decode("ascii"))

    key = cv.waitKey(1)
    if key == 27:  # ESC
        break
