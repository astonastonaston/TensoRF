import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np

from .ray_utils import *


class YourOwnDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [0.1,100.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):
        # load intrinsics
        K = np.loadtxt(os.path.join(self.root_dir, "intrinsics.txt"))

        w, h, focal = int(K[0, 2]*2), int(K[1, 2]*2), K[0, 0]
        self.img_wh = [w,h]
        self.focal_x = focal  # original focal length
        self.focal_y = focal  # original focal length
        self.cx, self.cy = K[0, 2], K[1, 2]

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        # load training and testing poses
        splits = ['train', 'val', 'test']
        split_sizes = [100, 100, 200]
        split_num = splits.index(self.split)
        assert self.split in splits
        split_size = split_sizes[split_num]
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        # self.all_masks = []
        # self.all_depth = []


        img_eval_interval = 1 if self.N_vis < 0 else split_size // self.N_vis
        idxs = list(range(0, split_size, img_eval_interval))
        is_test = False # at test state, no rgb is provided
        if self.split == "test": 
            is_test = True
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#
            # load poses
            posefname = os.path.join(self.root_dir, "pose", "{}_{}_{:04d}.txt".format(split_num, self.split, i))
            pose = np.array(np.loadtxt(posefname)) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            # load rgb images
            if not is_test:
                image_path = os.path.join(self.root_dir, "rgb", "{}_{}_{:04d}.png".format(split_num, self.split, i))
                self.image_paths += [image_path]
                img = Image.open(image_path)
                
                if self.downsample != 1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(-1, w*h).permute(1, 0)  # (h*w, 4) RGBA
                if img.shape[-1]==4:
                    img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                print(img.shape)
                self.all_rgbs += [img]

            # load world-frame rays
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)


        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            # mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img}
        return sample
