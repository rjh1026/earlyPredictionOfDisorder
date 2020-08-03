#--- Test and Visualize ---#
import time, datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from utils.visutil import *
from utils.osutils import import_module, join_path
from utils.logger import logger
from utils.config import get_config

#--- Global Vars ---#
#dirpath = join_path('pretrained_weights', 'stacked_hourglass', 'mpii')
dirpath = join_path('pretrained_weights', 'simple_baseline', 'mpii')
filepath = join_path(dirpath, 'test.yaml')
cfg = get_config(filepath)

# pretrained model
#ckpt_fname = 'hg_s2_b1.pth.tar'
ckpt_fname = 'pose_resnet_101_256x256.pth.tar'
ckpt_path = join_path(dirpath, ckpt_fname) # checkpoint path

# text logger
txt_logger = logger(filename=None)

# cuda / cpu device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
txt_logger.warning('Available Device:{}'.format(device))


### Parts Index (MPII 기준으로 indexing이 바뀝니다. pelvis, thorax가 추가되어 번호가 밀렸을뿐 다른 차이는 없습니다.)###
"""
1~3:r(ank,knee,hip)
4~6:l(hip,knee,ank)
(lsp) 7:neck, 8:head, 9~11:r(wrist,elbow,shoulder), 12~14:l(shoulder,elbow,wrist)
(mpii) 7:pelvis, 8:thorax, 9:neck, 10:head, 11~13:r(wrist,elbow,shoulder), 14~16:l(shoulder,elbow,wrist)
"""
# p_idx is the index of joints used to compute accuracy
if cfg.DATASET.NAME in ['lsp']:
    p_idx = [1,2,3,4,5,6,10,11,12,13,14,15,16] # pelvis(7), thorax(8), neck(9)을 제외한 나머지 parts의 정확도를 구함.
else:
    print("Unknown dataset: {}".format(cfg.DATASET.NAME))
    assert False


def main():
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    # load checkpoint
    checkpoint = torch.load(ckpt_path)
    #print(checkpoint.keys())

    ds_module = import_module('datasets.' + cfg.DATASET.NAME) # dataset path
    md_module = import_module('models.' + cfg.MODEL.NAME + '.model') # model path
    eval_module = import_module('evals.' + cfg.EVAL_METRIC) # evaluation metric path

    # get dataset
    test_dataset = ds_module.get_dataset(is_train=False, **cfg.DATASET)

    dataloader = { 'valid': DataLoader(test_dataset, batch_size=cfg.OPTS.BATCH_SIZE, 
                                        shuffle=False, num_workers=0, pin_memory=True) }

    # get model
    model = md_module.get_model(cfg)
    
    # stacked hg
    #model = torch.nn.DataParallel(model).to(device)
    #model.load_state_dict(checkpoint['state_dict'])
    
    # simple baseline
    model.load_state_dict(checkpoint)
    model = torch.nn.DataParallel(model).to(device)

    # model summary
    #summary(model.cuda(), input_size=(3, 256, 256), batch_size=-1)

    # loss
    from criterions.jointsmseloss import JointsMSELoss
    criterion = JointsMSELoss(use_target_weight=True).to(device)

    # Print current experiment setting
    txt_logger.warning('### Current Experiment Configs ###')
    txt_logger.warning(cfg)
    txt_logger.warning('##################################')
    t_s = time.time()

    # Test
    txt_logger.info('Testing...')
    test(model, dataloader['valid'], criterion, eval_module)

    # elapsed time
    t_e = time.time() - t_s
    times = str(datetime.timedelta(seconds=t_e)).split(".")[0]
    txt_logger.info('Training complete in {}'.format(times))


def test(model, dataloader, criterion, eval_module):
    
    model.eval()

    with torch.no_grad():
        cnt = 0
        vis_losses = []
        NUM_DATA = len(dataloader.dataset)
        TEST_VIS_STEP = torch.floor(torch.rand(3) * 100) # random samples

        idx = 0
        all_preds = torch.zeros((NUM_DATA, 16, 2))
        all_gts = torch.zeros((NUM_DATA, 16, 2))

        for i, (cropimg, inp, target, meta) in enumerate(tqdm(dataloader)):
            # send data to device
            inp, target = inp.to(device), target.to(device)

            # predicts
            outputs = model(inp)
            
            if i in TEST_VIS_STEP:
                show_img(cropimg[0])
                if type(outputs) == list:
                    show_heatmaps_all(outputs[-1][0].cpu())
                else:
                    show_heatmaps_all(outputs[0].cpu())
            
            # compute loss
            if type(outputs) == list: # multi-stage
                for s in range(len(outputs)):
                    loss = criterion(outputs[s], target, meta['target_weight'].to(device)) # compute all losses at the end of stack
                    
                    if len(vis_losses) != len(outputs):
                        vis_losses = [0] * len(outputs) # init
                    else:
                        vis_losses[s] += loss.item()

                score_map = outputs[-1]
            else: # single-stage
                loss = criterion(outputs, target, meta['target_weight'].to(device))

                if len(vis_losses) == 0:
                    vis_losses = [0] # init
                else:
                    vis_losses[0] += loss.item()

                score_map = outputs

            num_images = inp.size(0)
            preds = eval_module.get_preds(score_map) # heatmap으로부터 최대 값을 갖는 픽셀의 좌표를 얻습니다.
            gts = eval_module.get_preds(target) # BxPx2

            all_preds[idx:idx+num_images, :, 0:2] = preds[:, :, 0:2]
            all_gts[idx:idx+num_images, :, 0:2] = gts[:, :, 0:2]
            idx += num_images

            cnt += 1

        # compute accuracy using pck metric
        acc = eval_module.accuracy(all_preds, all_gts, p_idx, thr=0.5)

        # print all losses of stacks
        vis_losses = torch.tensor(vis_losses)
        vis_losses = vis_losses / cnt
        # stack이 하나인 single-stage인 경우 loss는 한번만 출력됩니다.
        loss_stack = {'stack' + str(stk+1): vis_losses[stk] for stk in range(vis_losses.size(0))} 
        print()
        print('loss_stack = ', loss_stack)
        avgloss = vis_losses.sum() / vis_losses.size(0)
        print('avgloss = ', avgloss)
        loss_stack.clear()

        # print accuracy
        acc_dict = {'acc[{}]'.format(idx): val for idx, val in enumerate(acc)}
        print()
        print(acc_dict)

        avgacc = acc[0]
        return avgloss, avgacc


if __name__ == '__main__':
    main()