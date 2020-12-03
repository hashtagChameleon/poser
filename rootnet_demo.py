import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

print(f'rootnet: {sys.path}')
sys.path.insert(0, osp.join('ROOTNET_RELEASE', 'main'))
sys.path.insert(0, osp.join('ROOTNET_RELEASE', 'data'))
sys.path.insert(0, osp.join('ROOTNET_RELEASE', 'common'))
print(f'rootnet: {sys.path}')
from config import cfg
from model import get_pose_net
from utils.pose_utils import process_bbox
from dataset import generate_patch_image
sys.path = sys.path[7:] # revert sys path to prevent colision with rootnet
print(f'rootnet: {sys.path}')

class Rootnet():
    def __init__(self, model_path, gpu_ids = '0'):
        cfg.set_args(gpu_ids)
        cudnn.benchmark = True

        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print(f'Loading Rootnet model {model_path}')
        self.model = DataParallel(get_pose_net(cfg, False)).cuda()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['network'])
        self.model.eval()

    def process_image(self, img_path, bbox_list):
        # prepare input image
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
        original_img = cv2.imread(img_path)
        original_img_height, original_img_width = original_img.shape[:2]

        person_num = len(bbox_list)

        # normalized camera intrinsics
        focal = [1500, 1500] # x-axis, y-axis
        princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
        print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
        print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

        # for cropped and resized human image, forward it to RootNet
        root_depth_list = []
        for n in range(person_num):
            bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0)
            img = transform(img).cuda()[None,:,:,:]
            k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
            k_value = torch.FloatTensor([k_value]).cuda()[None,:]

            # forward
            with torch.no_grad():
                root_3d = self.model(img, k_value) # x,y: pixel, z: root-relative depth (mm)
            img = img[0].cpu().numpy()
            root_3d = root_3d[0].cpu().numpy()

            root_depth_list.append(root_3d[2])

        return root_depth_list
