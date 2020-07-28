import os
import random
import json

import torch
import torchvision.transforms as tvtf
from torch.utils.data import Dataset

from codes.utils.osutils import *
from codes.utils.imutils import *
from codes.utils.transforms import *


def get_dataset(is_train=True, **kwargs):
    return LSP(is_train, **kwargs)


class LSP(Dataset):

    def __init__(self, is_train=True, **kwargs):
        self.img_folder = join_path('data', 'lsp')
        self.jsonfile = join_path('data', 'lsp', 'LEEDS_annotations.json')
        self.is_train = is_train
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma'] # 1
        
        # create train/val split
        with open(self.jsonfile) as anno_file:
            self.anno = json.load(anno_file)
        
        self.train_list = []
        self.valid_list = []

        for idx, obj in enumerate(self.anno):
            # pelvis(mid-hip), thorax(mid-shoulder)는 불필요하므로 주석에서 제거하였습니다.
            pts = obj['joint_self']
            del pts[6:8] # keypoint 16 -> 14

            # train set과 validation set을 나눕니다.
            if obj['isValidation'] == True:
                self.valid_list.append(idx)
            else:
                self.train_list.append(idx)

        self.mean, self.std = self._compute_mean()

        if self.is_train:
            self.sf = 0.25 # scale factor
            self.rf = 30   # rotate factor
            self.transform = tvtf.Compose([ColorJittering(0.8, 1.2), 
                                            ColorNormalize(self.mean, self.std)])
        else:
            self.transform = tvtf.Compose([ColorNormalize(self.mean, self.std)])

    # Normalize에 사용할 train set의 평균과 표준편차를 계산합니다.
    def _compute_mean(self):
        meanstd_file = join_path('data', 'lsp', 'mean.pth.tar') 

        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)

            # Compute mean and standard deviation of all train set images
            for idx in self.train_list:
                obj = self.anno[idx]
                img_path = join_path(self.img_folder, get_correct_path(obj['img_paths']))
                img = load_image(img_path)
                mean += img.view(img.size(0), -1).float().mean(1)
                std += img.view(img.size(0), -1).float().std(1)
            mean /= len(self.train_list)
            std /= len(self.train_list)
            meanstd = {
                'mean': mean,
                'std': std
            }
            torch.save(meanstd, meanstd_file) # 파일로 저장
        
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']


    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)


    def __getitem__(self, idx):
        if self.is_train:
            obj = self.anno[self.train_list[idx]]
        else:
            obj = self.anno[self.valid_list[idx]]

        img_path = join_path(self.img_folder, get_correct_path(obj['img_paths']))
        pts = torch.Tensor(obj['joint_self']) # keypoints 14 

        c = torch.Tensor(obj['objpos']) # rough human position in the image
        s = obj['scale_provided'] # person scale with respect to 200 px height

        # LSP uses matlab format, index is based 1,
        # we should first convert to 0-based index
        pts[:, 0:2] -= 1  
        c -= 1

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            #c[1] = c[1] + 15 * s
            s = s * 1.4375
        
        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path) # CxHxW

        # Transforms a image to feed the model's input
        r = 0
        if self.is_train:
            s = s * random.uniform(1 - self.sf, 1 + self.sf) # random scale
            r = random.uniform(-self.rf, self.rf) if random.random() <= 0.4 else 0 # random rotate

            # Flip
            if random.random() <= 0.5:
                img = fliplr_img(img)
                pts = fliplr_joints(pts, width=img.size(2), dataset='lsp')
                c[0] = img.size(2) - c[0]

        # scale, rotation, crop
        cropimg = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = self.transform(cropimg)

        # Generate ground truth
        tpts = pts.clone() # transformed points
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            if tpts[i, 2] > 0:
                tpts[i, 0:2] = to_torch(affine_transform(tpts[i, 0:2], c, s, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i], self.sigma) # gaussian ground truth image
                target_weight[i, 0] *= vis

        # Meta info
        meta = {'index' : idx, 'center' : c, 'scale' : s, 
        'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight}

        return cropimg, inp, target, meta
        