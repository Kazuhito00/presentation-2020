#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import eel
import cv2 as cv
import base64
import tensorflow as tf
import numpy as np

from boundingbox_art import *

current_slide = 0


@eel.expose
def slide_change_event(val):
    global current_slide
    current_slide = val


def main():
    global current_slide

    cap = cv.VideoCapture(0)

    # EeLフォルダ設定、および起動 ##################################################
    eel.init('webslides')
    eel.start(
        'index.html',
        mode='chrome',
        cmdline_args=['--start-fullscreen'],
        block=False)
    eel.sleep(1.0)

    # モデルロード ################################################################
    sess = graph_load('model/frozen_inference_graph.pb')
    eel.go_nextslide()

    while True:
        eel.sleep(0.01)

        # カメラキャプチャ ########################################################
        ret, frame = cap.read()
        if not ret:
            continue

        # スライド頁に応じた処理 ###################################################
        draw_image = image_processing(current_slide, frame, sess)

        # UI側へ転送 ##############################################################
        _, imencode_image = cv.imencode('.jpg', draw_image)
        base64_image = base64.b64encode(imencode_image)
        if current_slide == 1:
            eel.set_base64image01("data:image/jpg;base64," +
                                  base64_image.decode("ascii"))
        else:
            eel.set_base64image02("data:image/jpg;base64," +
                                  base64_image.decode("ascii"))


def image_processing(slide_number, image, sess=None):
    # print(slide_number)

    draw_image = copy.deepcopy(image)
    frame_width, frame_height = draw_image.shape[1], draw_image.shape[0]

    if slide_number == 1:
        is_detected = False

        inp = cv.resize(image, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        out = session_run(sess, inp)

        num_detections = int(out[0][0])
        for i in range(num_detections):
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            class_id = int(out[3][0][i])

            if score < 0.6 or class_id != 1:  # person
                continue

            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

            draw_image = bba_look_into_the_muzzle(
                image=draw_image,
                p1=(x1, y1),
                p2=(x2, y2),
            )

            is_detected = True
        if not is_detected:
            draw_image = copy.deepcopy(
                np.zeros((frame_height, frame_width, 3), np.uint8))
    else:
        inp = cv.resize(image, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        out = session_run(sess, inp)

        num_detections = int(out[0][0])
        for i in range(num_detections):
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            class_id = int(out[3][0][i])

            if score < 0.6 or class_id != 1:  # person
                continue

            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

            draw_image = bba_sound_only_monolith(
                image=draw_image,
                p1=(x1, y1),
                p2=(x2, y2),
                text='PYCON',
                number=i + 1,
            )

    return draw_image


def graph_load(path):
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

    with tf.compat.v1.Graph().as_default() as net_graph:
        graph_data = tf.gfile.FastGFile(path, 'rb').read()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(graph_data)
        tf.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=net_graph, config=config)
    sess.graph.as_default()

    return sess


def session_run(sess, inp):
    out = sess.run(
        [
            sess.graph.get_tensor_by_name('num_detections:0'),
            sess.graph.get_tensor_by_name('detection_scores:0'),
            sess.graph.get_tensor_by_name('detection_boxes:0'),
            sess.graph.get_tensor_by_name('detection_classes:0')
        ],
        feed_dict={
            'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)
        },
    )
    return out


if __name__ == '__main__':
    main()
