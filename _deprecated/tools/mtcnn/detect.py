""" A interface to MTCNN
    # (left_eye_x, left_eye_y)
    # (right_eye_x, right_eye_y)
    # (nose_x, nose_y)
    # (left_mouth_x, left_mouth_y)
    # (right_mouth_x, right_mouth_y)
    key_points = []

    # (left_top_x, left_top, y)
    # (right_bottom_x, right_bottom_y)
    # (width, height)
    key_boxs = []

    # confidence for face
    key_confidence = []
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import mtcnn
import tensorflow as tf
import numpy as np
# from PIL import Image


def print_all(points, boxs, confidences):
    for var in points:
        print(var)
    for var in boxs:
        print(var)
    for var in confidences:
        print(var)


def filter_by_confidence(points, boxs, confidences, threshold=0.9):
    _points = []
    _boxs = []
    _confidences = []
    for _i in range(len(points)):
        if confidences[_i] > threshold:
            _points.append(points[_i])
            _boxs.append(boxs[_i])
            _confidences.append(confidences[_i])
    return _points, _boxs, _confidences


def filter_by_num(points, boxs, confidences, num=1):
    if type(num) is not int:
        raise ValueError('num should be int value.')
    if num < 1:
        raise ValueError('num is equal and larger than 1.')
    _points = points[0:num]
    _boxs = boxs[0:num]
    _confidences = confidences[0:num]
    return _points, _boxs, _confidences


def draw(imgpath, points, boxs, confidences):
    points_x = []
    points_y = []
    for _i in range(len(points)):
        for _j in range(5):
            points_x.append(points[_i][_j][0])
            points_y.append(points[_i][_j][1])

    # bounding box width
    bb_w = []
    bb_h = []
    for _i in range(len(points)):
        bb_w.append(boxs[_i][1][0] - boxs[_i][0][0])
        bb_h.append(boxs[_i][1][1] - boxs[_i][0][1])

    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for _i in range(len(points)):
        ax.add_patch(patches.Rectangle(
            (boxs[_i][0][0], boxs[_i][0][1]),
            bb_w[_i], bb_h[_i],
            fill=False, linewidth=1, color='g'))

    plt.plot(points_x, points_y, ls='', marker='.', color='r', markersize=1)

    img = plt.imread(imgpath)
    # 1 inch = dpi pixels
    fig.set_size_inches(img.shape[1] / 300.0, img.shape[0] / 300.0)
    plt.imshow(img, aspect='auto')

    # plt.show()
    fig.savefig('test.11.jpg', dpi=300)


def detect(imgpath):
    with tf.Graph().as_default():

        img = plt.imread(imgpath)

        minsize = 20  # minimum size of face
        threshold = [0.9, 0.8, 0.8]  # three steps's threshold
        factor = 0.709  # scale factor

        key_points = []
        key_boxs = []
        key_confidence = []

        with tf.Session() as sess:
            pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)
            boxes, points = mtcnn.detect_face(
                img, minsize, pnet, rnet, onet, threshold, factor)

            num_people = points.shape[1]

            for _i in range(num_people):
                # key point
                _point = []
                for _j in range(5):
                    _point.append((points[_j][_i], points[5 + _j][_i]))
                key_points.append(_point)
                # box
                _box = [(boxes[_i][0], boxes[_i][1]),
                        (boxes[_i][2], boxes[_i][3]),
                        (boxes[_i][2] - boxes[_i][0],
                         boxes[_i][3] - boxes[_i][1])]
                key_boxs.append(_box)
                # confidence
                key_confidence.append(boxes[_i][4])

            return key_points, key_boxs, key_confidence


points, boxs, confidences = detect('test.jpg')
# points, boxs, confidences = filter_by_confidence(
#     points, boxs, confidences, 0.9)
points, boxs, confidences = filter_by_num(points, boxs, confidences, 1)
draw('test.jpg', points, boxs, confidences)
