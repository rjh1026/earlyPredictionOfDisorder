import math
import torch

from codes.utils.misc import *
from codes.utils.transforms import affine_transform, transform_preds


def get_preds(scores):
    ''' get a max value point in a score map

        각 part별 heatmap에서 max값이 0보다 큰 곳의 x, y 위치를 반환. max가 0보다 작으면 x, y는 0.

        return
        pred: (BxPx2) x, y
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2) # Bx16x4096에서 가장 높은 점수와 그 위치(0~4095)를 얻는다.

    maxval = maxval.view(scores.size(0), scores.size(1), 1) # Bx16 -> Bx16x1
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1 # 1~4096

    preds = idx.repeat(1, 1, 2).float() # Bx16x2

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1 # x= 1~64
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1 # y= 1~64

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float() # gt(input, 0): element-wise하게 input > 0을 비교하여 해당 element 위치에 True, False를 반환
                                                     # Bx16x2 (bool) -> Bx16x2 (float)
    preds *= pred_mask
    return preds # x, y = 1~64 or 0


def calc_dists(preds, target, normalize):
    """
    if x,y > 0, L2Norm distance
    else, -1
    
    return
    dists: distance or -1 (partxbatch)
    """
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0)) # parts x batch
    for b in range(preds.size(0)):
        for p in range(preds.size(1)):
            #if target[b, p, 0] > 1 and target[b, p, 1] > 1:
            if target[b, p, 0] > 0 and target[b, p, 1] > 0: # zero means there is no max value
                dists[p, b] = torch.dist(preds[b, p, :], target[b, p, 0:2])/normalize[b] # torch.dist()는 default로 L2norm distance를 구함
            else:
                dists[p, b] = -1
    return dists


def dist_acc(dist, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 

        return value (correct_num / predict_num)
    '''
    dist_cal = dist.ne(-1) # -1이 아니면 True
    num_dist_cal = dist_cal.sum()
    
    if num_dist_cal > 0:
        return 1.0 * (dist[dist_cal].lt(thr)).sum() / num_dist_cal # dist[[True, False, True]]이면, True인 인덱스의 요소만 출력
                                                                          # tensor.lt()는 a < b를 만족하는 요소를 True, False로 출력
    else:
        return -1

# PCK, PCKh
def accuracy(output, target, p_idxs, meta=None, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations.
        If meta['headsize'] is given, normalize distance by headsize according to PCKh.
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
        
        output: heatmap (BxPx64x64)
        target: heatmap (BxPx64x64)
        p_idxs: list
        meta: 'headsize', 'center', 'scale'

        return
        acc: tensor (parts + 1) [average acc, other parts, ...]
    '''

    # PCK
    if meta is None:
        preds = get_preds(output) # BxPx2
        gts = get_preds(target) # BxPx2
        norm = torch.ones(preds.size(0)) * output.size(3) / 10
    else: # PCKh
        preds = final_preds(output, meta['center'], meta['scale'], [64, 64]) # BxPx2
        #gts = final_preds(target, meta['center'], meta['scale'], [64, 64]) # BxPx2
        gts = meta['pts'] # 4, 16, 3
        gts = gts[:, :, 0:2] * gts[:, :, 2].unsqueeze(-1) # BxPx2 same result with the preds vars
        norm = meta['headsize']
        
    dists = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(p_idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(p_idxs)):
        acc[i+1] = dist_acc(dists[p_idxs[i]-1])
        if acc[i+1] >= 0:
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc


# 64x64 좌표 상의 예측한 좌표를 256x256 좌표로 변환
def final_preds(output, center, scale, res):
    coords = get_preds(output) # BxPx2 float type

    # pose-processing
    for b in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[b][p] # 64x64
            px = int(math.floor(coords[b][p][0]))
            py = int(math.floor(coords[b][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]: # px, py is 1~64
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], 
                                        hm[py][px - 1] - hm[py - 2][px - 1]]) # this means [hm[y][x+1]-hm[y][x-1], hm[y+1][x] - hm[y-1][x]]
                coords[b][p] += diff.sign() * .25 # +0.25 or -0.25
    coords += 0.5 # +0.5 - 0.25 or +0.5 + 0.25
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)): # batch
        preds[i] = transform_preds(coords[i], center[i], scale[i], res) # BxPx2

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds
