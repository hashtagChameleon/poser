import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('POSENET_RELEASE', 'main'))
sys.path.insert(0, osp.join('POSENET_RELEASE', 'data'))
sys.path.insert(0, osp.join('POSENET_RELEASE', 'common'))
from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
from utils.pose_utils import process_bbox, pixel2cam
from utils.vis import vis_keypoints, vis_3d_multiple_skeleton

model_path = '/home/levishai_g/pose_estimation/models/snapshot_24.pth.tar'

joint_num = 21
model = None

class Posenet():
    def __init__(self, model_path = model_path, gpu_ids = '0'):
        cfg.set_args(gpu_ids)
        cudnn.benchmark = True

        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print(f'Loading Posenet model {model_path}')
        self.model = DataParallel(get_pose_net(cfg, False, joint_num)).cuda()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['network'])
        self.model.eval()
        print(f'Model loaded succesfully')

    def process_image(self, original_img, bbox_list, root_depth_list):
        assert len(bbox_list) == len(root_depth_list)

        # prepare input image
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
        original_img_height, original_img_width = original_img.shape[:2]

        person_num = len(bbox_list)

        # normalized camera intrinsics
        focal = [1500, 1500] # x-axis, y-axis
        princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
        # print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
        # print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

        # for each cropped and resized human image, forward it to PoseNet
        output_pose_3d_list = []
        for n in range(person_num):
            bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False)
            img = transform(img).cuda()[None,:,:,:]

            # forward
            with torch.no_grad():
                pose_3d = self.model(img) # x,y: pixel, z: root-relative depth (mm)

            # inverse affine transform (restore the crop and resize)
            pose_3d = pose_3d[0].cpu().numpy()
            pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
            pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
            pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]

            # root-relative discretized depth -> absolute continuous depth
            pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth_list[n]
            pose_3d = pixel2cam(pose_3d, focal, princpt)
            output_pose_3d_list.append(pose_3d.copy())

        return output_pose_3d_list
