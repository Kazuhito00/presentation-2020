#!/usr/bin/env python
# -*- coding: utf-8 -*-

import eel
import cv2 as cv
import base64

##### ADD START #####
import copy
import tensorflow as tf
import numpy as np

from CvOverlayImage import CvOverlayImage
import FpsCalc


def session_run(sess, inp):
    out = sess.run([
        sess.graph.get_tensor_by_name('num_detections:0'),
        sess.graph.get_tensor_by_name('detection_scores:0'),
        sess.graph.get_tensor_by_name('detection_boxes:0'),
        sess.graph.get_tensor_by_name('detection_classes:0')
    ],
                   feed_dict={
                       'image_tensor:0':
                       inp.reshape(1, inp.shape[0], inp.shape[1], 3)
                   })
    return out


##### ADD END #####


@eel.expose
def slide_init_event():
    print("slide_init_event")


@eel.expose
def slide_change_event(val):
    print("slide_change_event:" + str(val))


def main():
    # EeLフォルダ設定、および起動 #########################################################
    eel.init('web')
    eel.start(
        'index.html',
        mode='chrome',
        # cmdline_args=['--start-fullscreen', '--browser-startup-dialog'])
        cmdline_args=['--start-fullscreen'],
        block=False)

    ##### ADD START #####
    print("Hand Detection Start...\n")

    # カメラ準備 ##############################################################
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    # FPS算出クラス起動 #######################################################
    fpsWithTick = FpsCalc.fpsWithTick()

    # GPUメモリを必要な分だけ確保
    # ※指定しない限りデフォルトではすべて確保する
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    # 手検出モデルロード #######################################################
    with tf.Graph().as_default() as net1_graph:
        graph_data = tf.gfile.FastGFile('frozen_inference_graph1.pb',
                                        'rb').read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_data)
        tf.import_graph_def(graph_def, name='')

    sess1 = tf.Session(graph=net1_graph, config=config)
    sess1.graph.as_default()

    animation_counter = 0
    ##### ADD END #####

    # cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        eel.sleep(0.01)

        # # カメラキャプチャ ########################################################
        # ret, frame = cap.read()
        # if not ret:
        #     continue

        animation_counter += 10
        # FPS算出 ####################################################
        display_fps = fpsWithTick.get()
        if display_fps == 0:
            display_fps = 0.1

        # カメラキャプチャ ###################################################
        ret, frame = cap.read()
        if not ret:
            continue
        debug_image = copy.deepcopy(frame)

        # 検出実施 ####################################################
        inp = cv.resize(frame, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        out = session_run(sess1, inp)

        rows = frame.shape[0]
        cols = frame.shape[1]

        # 検出結果可視化 ###############################################
        num_detections = int(out[0][0])
        for i in range(num_detections):
            class_id = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]

            if score < 0.8:
                continue

            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows

            radius = int((bottom - y) * (5 / 10))
            tickness = int(radius / 20)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 0 + animation_counter, 0,
                       50, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 80 + animation_counter,
                       0, 50, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 150 + animation_counter,
                       0, 30, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 200 + animation_counter,
                       0, 10, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 230 + animation_counter,
                       0, 10, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 260 + animation_counter,
                       0, 60, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 337 + animation_counter,
                       0, 5, (255, 255, 205), tickness)

            radius = int((bottom - y) * (4.5 / 10))
            tickness = int(radius / 10)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 0 - animation_counter, 0,
                       50, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 80 - animation_counter,
                       0, 50, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 150 - animation_counter,
                       0, 30, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 200 - animation_counter,
                       0, 30, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 260 - animation_counter,
                       0, 60, (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius), 337 - animation_counter,
                       0, 5, (255, 255, 205), tickness)

            radius = int((bottom - y) * (4 / 10))
            tickness = int(radius / 15)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius),
                       30 + int(animation_counter / 3 * 2), 0, 50,
                       (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius),
                       110 + int(animation_counter / 3 * 2), 0, 50,
                       (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius),
                       180 + int(animation_counter / 3 * 2), 0, 30,
                       (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius),
                       230 + int(animation_counter / 3 * 2), 0, 10,
                       (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius),
                       260 + int(animation_counter / 3 * 2), 0, 10,
                       (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius),
                       290 + int(animation_counter / 3 * 2), 0, 60,
                       (255, 255, 205), tickness)
            cv.ellipse(debug_image, (int((x + right) / 2), int(
                (y + bottom) / 2)), (radius, radius),
                       367 + int(animation_counter / 3 * 2), 0, 5,
                       (255, 255, 205), tickness)

        # UI側へ転送 ##############################################################
        # _, imencode_image = cv.imencode('.jpg', frame)
        _, imencode_image = cv.imencode('.jpg', debug_image)
        base64_image = base64.b64encode(imencode_image)
        eel.set_base64image("data:image/jpg;base64," +
                            base64_image.decode("ascii"))

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break


if __name__ == '__main__':
    main()
