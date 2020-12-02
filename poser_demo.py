import numpy
import json
from posenet_demo import Posenet
from datetime import datetime

joints_name2 = ('Head_top', 'Thorax', 'rightShoulder', 'rightElbow', 'rightWrist', 'leftShoulder', 'leftElbow', 'leftWrist', 'rightHip', 'rightKnee', 'rightAnkle', 'leftHip', 'leftKnee', 'leftAnkle', 'Pelvis', 'Spine', 'nose', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')

def now():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

posenet_model = '/content/drive/My Drive/Raven/3DMPPE/snapshot_24.pth.tar'
rootnet_model = '/content/drive/My Drive/Raven/3DMPPE/snapshot_18.pth.tar'

bbox_list = [
[162.12181091308594, 86.23428344726562, 203.85377502441406, 255.6412353515625],\
[286.82415771484375, 64.6770248413086, 74.55291748046875, 160.64566040039062],\
[539.04931640625, 49.81459045410156, 99.95068359375, 233.66294860839844],\
[365.950439453125, 170.007080078125, 270.9305419921875, 219.5510864257812],\
[0.7226638793945312, 44.89130401611328, 85.53158569335938, 221.61691284179688]]
root_depth_list = [11250.5732421875, 15522.8701171875, 11831.3828125, 8852.556640625, 12572.5966796875]

print(now())
posenet = Posenet(posenet_model)
print(now())
poses_3d = posenet.process_image('3DMPPE_POSENET_RELEASE/demo/input.jpg', bbox_list, root_depth_list)
print(now())
poses_3d = posenet.process_image('3DMPPE_POSENET_RELEASE/demo/input.jpg', bbox_list, root_depth_list)
print(now())

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

print(json.dumps({'timestamp': 0, 'poses': poses }))
