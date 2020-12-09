#!/usr/bin/env python3

import cv2
import numpy
import json
import requests
import pickle
import codecs
from functools import singledispatch
from datetime import datetime

from posenet_demo import Posenet

joints_name2 = ('Head_top', 'Thorax', 'rightShoulder', 'rightElbow', 'rightWrist', 'leftShoulder', 'leftElbow', 'leftWrist', 'rightHip', 'rightKnee', 'rightAnkle', 'leftHip', 'leftKnee', 'leftAnkle', 'Pelvis', 'Spine', 'nose', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')

def now():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

image = cv2.imread('/home/levishai_g/pose_estimation/images/input.jpg')

bbox_list = [
[162.12181091308594, 86.23428344726562, 203.85377502441406, 255.6412353515625],\
[286.82415771484375, 64.6770248413086, 74.55291748046875, 160.64566040039062],\
[539.04931640625, 49.81459045410156, 99.95068359375, 233.66294860839844],\
[365.950439453125, 170.007080078125, 270.9305419921875, 219.5510864257812],\
[0.7226638793945312, 44.89130401611328, 85.53158569335938, 221.61691284179688]]

print(type(pickle.dumps(image, protocol=0)))
rootnet_data = {
        'bbox_list': bbox_list,
        'image' : codecs.encode(pickle.dumps(image), "base64").decode() # protocol 0 is printable ASCII
        }
response = requests.post("http://localhost:3000", data=json.dumps(rootnet_data))
root_depth_list = json.loads(response.text)['root_depth_list']

posenet = Posenet()
poses_3d = posenet.process_image(image, bbox_list, root_depth_list)
poses = []

for pose in poses_3d:
   # print(pose)
   keypoints = []
   for i, joint in enumerate(pose):
       keypoint = {
           'score': 1,
           'part': joints_name2[i],
           'position' : {
               'x': joint[0],
               'y': joint[1],
               'z': joint[2],
           }
       }
       keypoints.append(keypoint)
       if joints_name2[i] == 'nose':
           keypoint = keypoint.copy()
           keypoint['part'] = 'leftEye'
           keypoints.append(keypoint)
           keypoint = keypoint.copy()
           keypoint['part'] = 'rightEye'
           keypoints.append(keypoint)
           keypoint = keypoint.copy()
           keypoint['part'] = 'leftEar'
           keypoints.append(keypoint)
           keypoint = keypoint.copy()
           keypoint['part'] = 'rightEar'
           keypoints.append(keypoint)

   poses.append({'score': 1, 'keypoints': keypoints})

print(json.dumps({'timestamp': 0, 'poses': poses }, default=to_serializable))
