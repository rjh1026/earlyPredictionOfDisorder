import random

import numpy as np
import cv2
import torch
import torchvision.transforms as tvtf

from utils.imutils import im_to_numpy, im_to_torch
from utils.misc import to_numpy, to_torch

# (from bearpaw)
def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m) # x[0]-mean[0]

    return x


class ColorNormalize():
    def __init__(self, mean, std, use_std=True):
        self.mean = mean
        self.std = std
        self.use_std = use_std

    def __call__(self, inp):
        out = inp.clone()

        if out.size(0) == 1:
            out = out.repeat(3, 1, 1)
        
        if self.use_std == True:
            out = tvtf.functional.normalize(out, self.mean, self.std)
        else:
            out[0] -= self.mean[0]
            out[1] -= self.mean[1]
            out[2] -= self.mean[2]

        return out


class ColorJittering():
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, inp):
        assert inp.dim() == 3
        out = inp.clone()

        out[0] *= random.uniform(self.low, self.high)
        out[1] *= random.uniform(self.low, self.high)
        out[2] *= random.uniform(self.low, self.high)
        
        return out.clamp(0, 1)


def crop(img, center, scale, res, rot=0):
    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]

    if sf < 2: # res / img scale 비율이 너무 작으면 그대로 진행
        sf = 1
    else: # res scale로 mapping할때 적절한 사람 크기가 되도록 resize
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            print('cannot cropping')
            return torch.zeros(res[0], res[1], img.shape[2]) if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else: # res scale 좌표계의 center와 scale을 구한다.
            img = cv2.resize(img, dsize=(new_wd, new_ht), interpolation=cv2.INTER_LINEAR) # cv2 img format: width x height
            center = center * 1.0 / sf
            scale = scale / sf # 1.28

    # Upper left point
    ul = np.array(affine_transform([0, 0], center, scale, res, invert=1))
    # invert가 1인 이유는 new img의 [0, 0] 좌표가 old img의 좌표 어디인지 알기위함.
    # Bottom right point
    br = np.array(affine_transform(res, center, scale, res, invert=1))

    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if rot != 0:
        ul -= pad
        br += pad

    # crop area
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape) # HWC

    # Range to fill new array (boundary zero area)
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if rot != 0:
        # Remove padding
        rmat = cv2.getRotationMatrix2D((new_shape[1]/2, new_shape[0]/2), rot, 1)
        new_img = cv2.warpAffine(new_img, rmat, (new_shape[1], new_shape[0]))
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(cv2.resize(new_img, dsize=(res[1], res[0]), interpolation=cv2.INTER_LINEAR))
    return new_img


# Helper functions


def flip_back(flip_output, dataset='mpii'):
    """
    flip output map
    """
    if dataset in ['lsp']:
        matchedParts = (
            [0,5],  [1,4], [2,3],
            [8,13], [9,12], [10,11]
        )
    elif dataset in ['mpii']:
        matchedParts = (
            [0,5],  [1,4], [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)
        assert False

    # flip output horizontally
    flip_output = fliplr_img(flip_output.numpy())

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()

# flip joints
def fliplr_joints(x, width, dataset='lsp'):
    """
    flip coords
    """
    if dataset in ['lsp']:
        matchedParts = (
            [0,5],  [1,4], [2,3],
            [8,13], [9,12], [10,11]
        )
    elif dataset in ['mpii']:
        matchedParts = (
            [0,5],  [1,4], [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)
        assert False

    # Flip horizontal
    x = x.clone()
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x

# flip image
def fliplr_img(x):
    x = x.numpy() # tensor to numpy

    if x.ndim == 3:
        x = np.transpose(x, (0, 2, 1))
        x = np.fliplr(x)               # fliplr() 설명과는 다르게 왜인지는 모르겠지만 3차원 텐서는 left right이 아니라 up-down이 바뀐다.
        x = np.transpose(x, (0, 2, 1)) # 결과적으로 left-right이 바뀌게 됨.
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
            print('warning')

    return torch.from_numpy(x.astype(float)).float() # return tensor

# Generate transformation matrix (old res coord - > new res coord)
def get_affine_transform(center, scale, res, rot=0):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h # old scale -> new scale
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5) # old -> new 로 이동시키는 translation distance (new center - old center)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1

    if rot != 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))

    return t


def affine_transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_affine_transform(center, scale, res, rot=rot)
    if invert: # new coord -> old coord
        t = np.linalg.inv(t)
    #new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt) # matrix multiplication
    return new_pt[:2].astype(int)

# Transform back to the original(old) coords
def transform_preds(coords, center, scale, res):
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(affine_transform(coords[p, 0:2], center, scale, res, invert=1))
    return coords


### ref ###

"""
def transform_preds(coords, center, scale, res):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, res, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, res, rot=0, 
        shift=np.array([0, 0], dtype=np.float32), inv=0):

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    center = to_numpy(center)

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = res[0]
    dst_h = res[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, center, scale, res, rot=0, inv=0):
    # Transform pixel location to different reference
    t = get_affine_transform(center, scale, res, rot=rot, inv=inv)

    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2] # .astype(int)


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, res, rot=0):
    img = im_to_numpy(img)
    trans = get_affine_transform(center, scale, res, rot=rot)

    dst_img = cv2.warpAffine(
        img, trans, (int(res[0]), int(res[1])),
        flags=cv2.INTER_LINEAR
    )

    return im_to_torch(dst_img)
"""