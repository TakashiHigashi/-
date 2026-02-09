import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import time
from einops import rearrange, repeat
import sys
from colour import Color
import pickle

def get_color_map(num_colors = 5):
    red = Color('red')
    blue = Color('blue')
    # 5分割 [<Color red>, <Color yellow>, <Color lime>, <Color cyan>, <Color blue>]
    red_blue = list(red.range_to(blue, num_colors))
    return red_blue
    


class RGBDDataset(Dataset):
    def __init__(self, txt_path, total_frames=80, type='rgb'):
        self.file_paths = []
        self.labels = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                path, label = line.rstrip('\n').split(' ')
                self.file_paths.append(path)
                self.labels.append(int(label))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.total_frames = total_frames
        self.get_frames = self.get_frames_avi if type=='rgb' else self.get_frames_png
        
        match type:
            case 'rgb':
                self.ALIGN = self.align_frames
                self.GET_FRAMES = self.get_frames_avi
            case 'depth':
                self.ALIGN = self.align_frames
                self.GET_FRAMES = self.get_frames_png
            case 'skeleton':
                self.ALIGN = self.align_frames
                self.GET_FRAMES = self.get_frames_skeleton
        
            
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        label = self.labels[idx]
        frames = self.GET_FRAMES(video_path)
        movie = self.ALIGN(frames)
        return movie, label
    
    def get_frames_avi(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return frames
    
    def get_frames_png(self, dir_path):
        img_paths = sorted(glob.glob(os.path.join(dir_path, '*.png')))
        frames = []
        for path in img_paths:
            img = Image.open(path)
            color = self.convert_depth_to_rgbd(img)
            frame = self.transform(color)
            frames.append(frame)
        return frames
    
    def align_frames(self, frames):
        num_frames = len(frames)
        if num_frames < self.total_frames:
            # フレームが少ない場合、最初のフレームを前に追加して補完
            # first_frame = frames[0]
            # front_padding = [first_frame] * (self.total_frames - num_frames)
            # frames = front_padding + frames
            pass
        else:
            # フレームが多い場合、等間隔で間引く
            indices = np.round(np.linspace(0, num_frames - 1, self.total_frames)).astype(int)
            frames = [frames[i] for i in indices]

        movie = torch.stack(frames)
        return movie
    
    def convert_depth_to_rgbd(self, depth_image):
        depth_image = np.array(depth_image)
        # print(depth_image_.max(), depth_image_.min(), depth_image_.mean())
        # depth_normalized = (np.array(depth_image) / 65535.0).astype(np.float32)
        depth_normalized = ((depth_image-depth_image.min()) / (depth_image.max()-depth_image.min())).astype(np.float32)
        depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        rgbd_image = Image.fromarray(depth_colormap)
        return rgbd_image
    
    def get_unique_colors(self, num_colors):
        np.random.seed(0)
        colors = np.random.rand(num_colors, 3) * 255  
        return colors.astype(int)

    def get_frames_skeleton(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        frames = list(data.values())
        num_persons = max([len(d.keys()) for d in frames])
        colors = get_color_map(num_persons)
        
        images = []
        for f in range(len(frames)):
            bg = np.zeros((224, 224, 3), dtype=np.uint8)  # 3チャンネルのRGB画像を使用
            for i, person_id in enumerate(frames[f]):  # 各人物のインデックスを取得
                person_data = frames[f][person_id]
                coordinates = np.array(((person_data[:, :2] + 1) / 2) * 224, dtype=int)
                coordinates[:, 1] = 224 - coordinates[:, 1]  # y座標を反転
                color = np.array(colors[i].get_rgb()) * 255  # 色情報を取得

                for coord in coordinates:
                    x, y = coord
                    cv2.circle(bg, (x, y), 2, color, -1)  # 各点を描画

                for _, (start_idx, end_idx) in line_info_num.items():
                    start_coord = coordinates[start_idx]
                    end_coord = coordinates[end_idx]
                    cv2.line(bg, tuple(start_coord), tuple(end_coord), color, 5)  # 各線を描画
            pil_img = Image.fromarray(bg)
            img = self.transform(pil_img)
            images.append(img)
        frames = images
        return frames
    
        
    def align_frames_skeleton(self, frames):
        num_frames, n_id, n_joint, dim = frames.shape
        
        if num_frames < self.total_frames:
            # フレームが少ない場合、最後のフレームを繰り返して追加
            last_frame = frames[:, -1:]
            back_padding = last_frame.repeat(1, self.total_frames - num_frames, 1, 1)
            frames = torch.cat([frames, back_padding], dim=1)
            
        elif num_frames > self.total_frames:
            # フレームが多い場合、等間隔で選択
            indices = torch.linspace(0, num_frames - 1, self.total_frames).to(torch.long)
            frames = frames[:, indices]
        
        # フレーム数が total_frames になるまで調整
        while frames.shape[1] > self.total_frames:
            frames = frames[:, 1:]  # 最初のフレームを削除
        while frames.shape[1] < self.total_frames:
            frames = torch.cat([frames, frames[:, -1:]], dim=1)  # 最後のフレームを追加
        
        assert frames.shape[1] == self.total_frames, f"Expected {self.total_frames} frames, but got {frames.shape[1]}"
        return frames

def custom_collate(batch):
    frames, targets = list(zip(*batch))
    targets = torch.tensor(targets)
    return list(frames), targets


# def custom_collate(batch):
#     frames, targets = list(zip(*batch))
#     # 各フレームのサイズが同じなのでテンソルスタックが可能
#     frames = torch.stack(frames)
#     targets = torch.tensor(targets)
#     return frames, targets

joint_info = {
    'SPINEBASE':0,
    'SPINEMID':1,
    'NECK':2,
    'HEAD':3,
    'SHOULDERLEFT':4,
    'ELBOWLEFT':5,
    'WRISTLEFT':6,
    'HENDLEFT':7,
    'SHOULDERRIGHT':8,
    'ELBOWRIGHT':9,
    'WRISTRIGHT':10,
    'HENDRIGHT':11,
    'HIPLEFT':12,
    'KNEELEFT':13,
    'ANKLELEFT':14,
    'FOOTLEFT':15,
    'HIPRIGHT':16,
    'KNEERIGHT':17,
    'ANKLERIGHT':18,
    'FOOTRIGHT':19,
    'SPINSHOULDER':20,
    'HANDTIPLEFT':21,
    'THUMBLEFT':22,
    'HANDTIPRIGHT':23,
    'THUMBRIGHT':24,
}

line_info = [
    ('HEAD','NECK'),
    ('NECK','SPINSHOULDER'),
    ('SPINSHOULDER','SHOULDERLEFT'),
    ('SHOULDERLEFT','ELBOWLEFT'),
    ('ELBOWLEFT','WRISTLEFT'),
    ('WRISTLEFT','HENDLEFT'),
    ('HENDLEFT','HANDTIPLEFT'),
    ('HENDLEFT','THUMBLEFT'),
    ('SPINSHOULDER','SHOULDERRIGHT'),
    ('SHOULDERRIGHT','ELBOWRIGHT'),
    ('ELBOWRIGHT','WRISTRIGHT'),
    ('WRISTRIGHT','HENDRIGHT'),
    ('HENDRIGHT','HANDTIPRIGHT'),
    ('HENDRIGHT','THUMBRIGHT'),
    ('SPINSHOULDER','SPINEMID'),
    ('SPINEMID','SPINEBASE'),
    ('SPINEBASE','HIPLEFT'),
    ('SPINEBASE','HIPRIGHT'),
    ('HIPLEFT','KNEELEFT'),
    ('KNEELEFT','ANKLELEFT'),
    ('ANKLELEFT','FOOTLEFT'),
    ('HIPRIGHT','KNEERIGHT'),
    ('KNEERIGHT','ANKLERIGHT'),
    ('ANKLERIGHT','FOOTRIGHT'),
]

line_info_num = {i:tuple(map(lambda x:joint_info[x],pear)) for i,pear in enumerate(line_info)}