#!/usr/bin/env python3

import cv2
import numpy as np
import json
import requests
import pickle
import codecs
import argparse
from functools import singledispatch
from datetime import datetime

from detectron_demo import Detectron
from posenet_demo import Posenet

joints_name2 = ('Head_top', 'Thorax', 'rightShoulder', 'rightElbow', 'rightWrist', 'leftShoulder', 'leftElbow', 'leftWrist', 'rightHip', 'rightKnee', 'rightAnkle', 'leftHip', 'leftKnee', 'leftAnkle', 'Pelvis', 'Spine', 'nose', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Poser')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-v', '--video', dest='videoFilePath', help='video input file', required=True)
    requiredNamed.add_argument('-o', '--output', dest='outputRecordingPath', help='output recording file', required=True)
    args = parser.parse_args()

    recordingFile = open(args.outputRecordingPath, 'w')
    # image = cv2.imread('/home/levishai_g/pose_estimation/images/input.jpg')
    cap = cv2.VideoCapture(args.videoFilePath)
    videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    currentFrame = 0

    detectron = Detectron()
    posenet = Posenet()

    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            currentFrame = currentFrame + 1
        else:
            print(f'failed to extract frame {currentFrame} from the video')
            break

        print(f'===== Processing frame # {currentFrame} / {videoFrames} =====')

        print('detectron running')
        bbox_list = detectron.process_image(image)
        print('rootnet running')
        rootnet_request_data = {
            'bbox_list': bbox_list,
            'image' : codecs.encode(pickle.dumps(image), "base64").decode() # protocol 0 is printable ASCII
            }
        response = requests.post("http://localhost:3000", data=json.dumps(rootnet_request_data, cls=NumpyEncoder))
        root_depth_list = json.loads(response.text)['root_depth_list']

        print('posenet running')
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

        recordingFile.write(json.dumps({
                'timestamp': int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                'poses': poses },
            cls=NumpyEncoder))
        recordingFile.write('\n')

    cap.release()
    recordingFile.close()
