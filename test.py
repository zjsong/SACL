import os
import cv2
import json
import random
import warnings
import numpy as np
from sklearn.metrics import auc

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from models import ModelBuilder
from dataset.videodataset import VideoDataset

from arguments_test import ArgParser
from utils import makedirs, AverageMeter, save_visual, normalize_img, normalize_tensor_imgs, testset_gt

warnings.filterwarnings('ignore')


def main():
    # arguments
    parser = ArgParser()
    args = parser.parse_arguments()

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.epoch_iters = args.num_train // args.batch_size
    args.device = torch.device("cuda")
    args.vis = os.path.join(args.ckpt, 'vis/')

    args.world_size = args.num_gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'xxx.xx.xx.xx'  # specified by yourself
    os.environ['MASTER_PORT'] = '8899'  # specified by yourself
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,))


def main_worker(gpu, args):
    rank = args.nr * args.num_gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ################################
    # model
    ################################
    builder = ModelBuilder()
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        train_from_scratch=args.train_from_scratch,
        fine_tune=args.fine_tune,
        weights_resnet_imgnet=args.weights_resnet_imgnet
    )
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        weights_vggish=args.weights_vggish,
        out_dim=args.out_dim
    )

    torch.cuda.set_device(gpu)

    net_frame = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_frame).cuda(gpu)
    net_sound = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_sound).cuda(gpu)

    # wrap model
    netWrapper = NetWrapper(net_frame, net_sound, args)
    netWrapper = torch.nn.parallel.DistributedDataParallel(netWrapper, device_ids=[gpu], find_unused_parameters=True)
    netWrapper.to(args.device)

    # load well-trained model
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    net_frame.load_state_dict(torch.load(args.weights_frame, map_location=map_location))
    net_sound.load_state_dict(torch.load(args.weights_sound, map_location=map_location))

    ################################
    # data
    ################################
    dataset_test = VideoDataset(args, mode='test')
    args.batch_size_ = int(args.batch_size / args.num_gpus)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    # gt for vggss
    if args.testset == 'vggss':
        args.gt_all = {}
        gt_path = args.data_path + 'VGG-Sound/5k_labeled/Annotations/vggss_test_5158.json'
        with open(gt_path) as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            args.gt_all[annotation['file']] = annotation['bbox']

    ################################
    # evaluation
    ################################
    if gpu == 0:

        evaluate(netWrapper, loader_test, args)

        print('Evaluation done!')


class NetWrapper(torch.nn.Module):
    def __init__(self, net_frame, net_sound, args):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net_sound = net_frame, net_sound
        self.tempD = args.tempD
        self.ratio_neg = args.ratio_neg
        self.ratio_retain_sim = args.ratio_retain_sim

    def forward(self, batch_data):

        image_view1 = batch_data['frame_view1']
        image_view2 = batch_data['frame_view2']
        mask_view1 = batch_data['mask_view1']
        mask_view2 = batch_data['mask_view2']
        spect = batch_data['spect']

        image_view1 = image_view1.cuda(non_blocking=True)
        image_view2 = image_view2.cuda(non_blocking=True)
        mask_view1 = mask_view1.cuda(non_blocking=True)
        mask_view2 = mask_view2.cuda(non_blocking=True)
        spect = spect.cuda(non_blocking=True)

        B = image_view1.size(0)

        # 1. forward net_sound
        audio_feat_orig = []
        audio_feat_trans = []
        for i in range(B):
            audio_feat_orig_i, audio_feat_trans_i = self.net_sound(spect[i])
            audio_feat_orig.append(audio_feat_orig_i.mean(dim=0))
            audio_feat_trans.append(audio_feat_trans_i.mean(dim=0))
        audio_feat_orig = torch.stack(audio_feat_orig, dim=0)
        audio_feat_orig = F.normalize(audio_feat_orig, p=2, dim=1)
        audio_feat_trans = torch.stack(audio_feat_trans, dim=0)

        # 2. forward net_frame
        # view 1
        vis_feat_view1 = self.net_frame(image_view1)
        _, sim_map_view1, _ = self.att_map_weight(vis_feat_view1, audio_feat_trans)

        # view 2
        vis_feat_view2 = self.net_frame(image_view2)
        _, sim_map_view2, _ = self.att_map_weight(vis_feat_view2, audio_feat_trans)

        # 3. feature selection
        # 3.1 use audio feature similarity to compute mask for true negative selection
        audio_feat_sim = torch.matmul(audio_feat_orig, audio_feat_orig.transpose(0, 1))  # B x B
        sort_sim = torch.argsort(audio_feat_sim, dim=1)  # indices, arranging values from smallest to largest
        num_neg = int(self.ratio_neg * B)
        select_neg_mask = torch.eye(B, B).type_as(vis_feat_view1)  # B x B
        select_neg_mask = select_neg_mask.scatter_(1, sort_sim[:, :num_neg], 1)

        # 3.2 masks for positive and negative pixels
        binary_sim_map_view1, binary_sim_map_view2, \
        contr_mask_pos_view1, contr_mask_pos_view2, \
        contr_mask_neg_view1, contr_mask_neg_view2 = self.pixel_contrast(sim_map_view1, sim_map_view2,
                                                                         mask_view1, mask_view2)

        # 3.3 compact visual features
        vis_feat_view1 = F.normalize(vis_feat_view1, p=2, dim=1)  # B x C x H x W
        vis_feat_view2 = F.normalize(vis_feat_view2, p=2, dim=1)  # B x C x H x W
        mask_pos_view1 = vis_feat_view1 * contr_mask_pos_view1.unsqueeze(1)  # B x C x H x W
        mask_pos_view2 = vis_feat_view2 * contr_mask_pos_view2.unsqueeze(1)  # B x C x H x W

        audio_feat_trans = F.normalize(audio_feat_trans, p=2, dim=1)  # B x C

        # 4 modulated InfoNCE loss
        loss = self.mask_pos_infonce(mask_pos_view1, mask_pos_view2, audio_feat_trans, select_neg_mask)

        output = {'orig_sim_map': sim_map_view1}

        return loss, output

    def att_map_weight(self, vis_feat_map, audio_feat_vec):

        # normalize visual feature
        B, C, H, W = vis_feat_map.size()
        vis_feat_map_trans = F.normalize(vis_feat_map, p=2, dim=1)
        vis_feat_map_trans = vis_feat_map_trans.view(B, C, H * W)
        vis_feat_map_trans = vis_feat_map_trans.permute(0, 2, 1).contiguous()  # B x (HW) x C

        # normalize audio feature
        audio_feat_vec = F.normalize(audio_feat_vec, p=2, dim=1)
        audio_feat_vec = audio_feat_vec.unsqueeze(2)  # B x C x 1

        # similarity/attention map
        att_map_orig = torch.matmul(vis_feat_map_trans, audio_feat_vec)  # B x (HW) x 1

        # min-max normalization on similarity map
        att_map = torch.squeeze(att_map_orig)  # B x (HW)
        att_map = (att_map - torch.min(att_map, dim=1, keepdim=True).values) / \
                  (torch.max(att_map, dim=1, keepdim=True).values - torch.min(att_map, dim=1,
                                                                              keepdim=True).values + 1e-10)
        att_map = att_map.view(B, 1, H, W)

        av_feat = vis_feat_map * att_map

        return av_feat, att_map_orig.view(B, H, W), att_map.view(B, H, W)

    def pixel_contrast(self, sim_map_view1, sim_map_view2, mask_view1, mask_view2):

        B, H, W = sim_map_view1.size()

        #################################################################
        # split pseudo segmentation mask into several binary masks
        #################################################################
        mask_view1 = F.interpolate(mask_view1, size=(H, W), mode='nearest')  # B x H x W
        mask_view2 = F.interpolate(mask_view2, size=(H, W), mode='nearest')  # B x H x W

        binary_masks_view1 = [0] * B
        binary_masks_view2 = [0] * B
        for i in range(B):
            if mask_view1[i].max() == 0 or mask_view2[i].max() == 0:   # only one pseudo mask
                binary_masks_view1[i] = torch.ones_like(sim_map_view1[i]).unsqueeze(0)   # 1 x H x W
                binary_masks_view2[i] = torch.ones_like(sim_map_view2[i]).unsqueeze(0)   # 1 x H x W

            else:   # at least two pseudo masks

                obj_ids_view1 = torch.unique(mask_view1[i])
                obj_ids_view2 = torch.unique(mask_view2[i])

                ids_pos_view1 = torch.zeros((torch.max(obj_ids_view1.max(), obj_ids_view2.max()) + 1,)).to(
                    mask_view1.device)
                ids_pos_view2 = torch.zeros((torch.max(obj_ids_view1.max(), obj_ids_view2.max()) + 1,)).to(
                    mask_view2.device)
                ids_pos_view1[obj_ids_view1.type(torch.int64)] = 1
                ids_pos_view2[obj_ids_view2.type(torch.int64)] = 1

                ids_shared_two_views = ids_pos_view1 * ids_pos_view2
                ids_shared_two_views = torch.where(ids_shared_two_views == 1)[0]

                binary_masks_view1_i = torch.zeros(len(ids_shared_two_views), H, W).to(mask_view1.device)
                binary_masks_view2_i = torch.zeros(len(ids_shared_two_views), H, W).to(mask_view2.device)
                for j, id in enumerate(ids_shared_two_views):
                    binary_masks_view1_i[j, :, :] = torch.where(mask_view1[i] == id, torch.tensor(1).to(mask_view1.device),
                                                                torch.tensor(0).to(mask_view1.device))
                    binary_masks_view2_i[j, :, :] = torch.where(mask_view2[i] == id, torch.tensor(1).to(mask_view2.device),
                                                                torch.tensor(0).to(mask_view2.device))

                binary_masks_view1[i] = binary_masks_view1_i
                binary_masks_view2[i] = binary_masks_view2_i

        #################################################################
        # compute contrast mask with the largest intersection area
        #################################################################
        norm_sim_map_view1 = normalize_tensor_imgs(sim_map_view1)  # B x H x W
        binary_sim_map_view1 = norm_sim_map_view1.clone().view(B, -1)
        threshold_view1 = torch.sort(binary_sim_map_view1, dim=-1).values[:, int(H * W * self.ratio_retain_sim)]  # B
        binary_sim_map_view1[binary_sim_map_view1 > threshold_view1.unsqueeze(1)] = 1
        binary_sim_map_view1[binary_sim_map_view1 < 1] = 0
        binary_sim_map_view1 = binary_sim_map_view1.view_as(sim_map_view1)

        norm_sim_map_view2 = normalize_tensor_imgs(sim_map_view2)  # B x H x W
        binary_sim_map_view2 = norm_sim_map_view2.clone().view(B, -1)
        threshold_view2 = torch.sort(binary_sim_map_view2, dim=-1).values[:, int(H * W * self.ratio_retain_sim)]  # B
        binary_sim_map_view2[binary_sim_map_view2 > threshold_view2.unsqueeze(1)] = 1
        binary_sim_map_view2[binary_sim_map_view2 < 1] = 0
        binary_sim_map_view2 = binary_sim_map_view2.view_as(sim_map_view2)  # B x H x W

        contr_mask_pos_view1 = torch.zeros_like(sim_map_view1)  # B x H x W
        contr_mask_pos_view2 = torch.zeros_like(sim_map_view2)  # B x H x W
        contr_mask_neg_view1 = torch.zeros_like(sim_map_view1)  # B x H x W
        contr_mask_neg_view2 = torch.zeros_like(sim_map_view2)  # B x H x W
        for i in range(B):

            if len(binary_masks_view1[i]) == 0 or len(binary_masks_view2[i]) == 0:
                contr_mask_pos_view1[i] = binary_sim_map_view1[i]
                contr_mask_neg_view1[i] = 1 - binary_sim_map_view1[i]

                contr_mask_pos_view2[i] = binary_sim_map_view2[i]
                contr_mask_neg_view2[i] = 1 - binary_sim_map_view2[i]
            else:
                # view 1
                sim_mul_mask_view1_i = torch.mul(binary_sim_map_view1[i], binary_masks_view1[i])  # num_obj x H x W
                sum_sim_mul_mask_view1_i = torch.sum(sim_mul_mask_view1_i, (1, 2))   # num_obj
                contr_mask_id_view1_i = torch.argmax(sum_sim_mul_mask_view1_i)  # mask with the largest intersection area
                contr_mask_pos_view1[i] = sim_mul_mask_view1_i[contr_mask_id_view1_i]
                contr_mask_neg_view1[i] = 1 - binary_masks_view1[i][contr_mask_id_view1_i]

                # view 2
                sim_mul_mask_view2_i = torch.mul(binary_sim_map_view2[i], binary_masks_view2[i])  # num_obj x H x W
                # same id for view 1 and view 2
                if torch.sum(sim_mul_mask_view2_i[contr_mask_id_view1_i]) == 0:
                    contr_mask_pos_view2[i] = binary_masks_view2[i][contr_mask_id_view1_i]  # H x W
                else:
                    contr_mask_pos_view2[i] = sim_mul_mask_view2_i[contr_mask_id_view1_i]  # H x W
                contr_mask_neg_view2[i] = 1 - binary_masks_view2[i][contr_mask_id_view1_i]

        return binary_sim_map_view1, binary_sim_map_view2, \
               contr_mask_pos_view1, contr_mask_pos_view2, \
               contr_mask_neg_view1, contr_mask_neg_view2

    def mask_pos_infonce(self, mask_pos_view1, mask_pos_view2, audio_feat_vec, select_neg_mask=None):

        B, C, H, W = mask_pos_view1.size()
        if select_neg_mask is not None:
            false_negative_mask = 1 - select_neg_mask    # false negatives
        else:
            false_negative_mask = (~torch.eye(B, B, dtype=bool)).cuda(non_blocking=True)  # all other samples are negatives

        # similarity between visual feature and all audio features
        mask_pos_view1 = mask_pos_view1.view(B, C, -1)   # B_v x C x (HW)
        mask_pos_view1 = mask_pos_view1.permute(2, 0, 1).contiguous()   # (HW) x B_v x C
        mask_pos_view2 = mask_pos_view2.view(B, C, -1)  # B_v x C x (HW)
        mask_pos_view2 = mask_pos_view2.permute(2, 0, 1).contiguous()  # (HW) x B_v x C

        audio_feat_vec = torch.transpose(audio_feat_vec, 0, 1)  # C x B_a

        sim_view1 = torch.matmul(mask_pos_view1, audio_feat_vec)  # (HW) x B_v x B_a
        sim_view1 = torch.max(sim_view1, dim=0).values.squeeze()  # B_v x B_a
        sim_view2 = torch.matmul(mask_pos_view2, audio_feat_vec)  # (HW) x B_v x B_a
        sim_view2 = torch.max(sim_view2, dim=0).values.squeeze()  # B_v x B_a

        # cross entropy
        labels = torch.arange(B).long().to(sim_view1.device)
        sim_view1[false_negative_mask.bool()] = -10   # make the effect of false negatives on loss negligible
        sim_view2[false_negative_mask.bool()] = -10   # make the effect of false negatives on loss negligible
        loss = F.cross_entropy(sim_view1 / self.tempD, labels) + F.cross_entropy(sim_view2 / self.tempD, labels)

        return loss.mean()


def evaluate(netWrapper, loader, args):
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    netWrapper.eval()

    loss_meter = AverageMeter()
    ciou_orig_sim = []
    for i, batch_data in enumerate(loader):

        with torch.no_grad():
            loss, output = netWrapper.forward(batch_data)

            img = batch_data['frame_view1'].numpy()
            sound = batch_data['sound'].numpy()
            video_id = batch_data['data_id']

            orig_sim_map = output['orig_sim_map'].detach().cpu().numpy()

            # convert similarity map into object mask and save for visualization
            for n in range(img.shape[0]):

                gt_map = testset_gt(args, video_id[n])
                ciou, _, _ = eval_cal_ciou(orig_sim_map[n], gt_map, img_size=args.imgSize, thres=0.5)
                ciou_orig_sim.append(ciou)

                save_visual(video_id[n], img[n], sound[n], orig_sim_map[n], args)

        loss_meter.update(loss.item())
        if args.testset == "flickr":
            print('[Eval] iter {}, loss: {}'.format(i, loss.item()))

        elif args.testset == "vggss" and i % 36 == 0:
            print('[Eval] iter {}, loss: {}'.format(i, loss.item()))

    # compute cIoU and AUC on whole dataset
    results_orig_sim = []
    for i in range(21):
        result_orig_sim = np.sum(np.array(ciou_orig_sim) >= 0.05 * i)
        result_orig_sim = result_orig_sim / len(ciou_orig_sim)
        results_orig_sim.append(result_orig_sim)

    x = [0.05 * i for i in range(21)]
    cIoU_orig_sim = np.sum(np.array(ciou_orig_sim) >= 0.5) / len(ciou_orig_sim)
    AUC_orig_sim = auc(x, results_orig_sim)

    print('[Eval Summary] Loss: {:.4f}, cIoU_orig_sim: {:.4f}, AUC_orig_sim: {:.4f}'.format(
        loss_meter.average(), cIoU_orig_sim, AUC_orig_sim))


def eval_cal_ciou(heat_map, gt_map, img_size=224, thres=None):

    # preprocess heatmap
    heat_map = cv2.resize(heat_map, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    heat_map = normalize_img(heat_map)

    # convert heatmap to mask
    pred_map = heat_map
    if thres is None:
        threshold = np.sort(pred_map.flatten())[int(pred_map.shape[0] * pred_map.shape[1] / 2)]
        pred_map[pred_map >= threshold] = 1
        pred_map[pred_map < 1] = 0
        infer_map = pred_map
    else:
        infer_map = np.zeros((img_size, img_size))
        infer_map[pred_map >= thres] = 1

    # compute ciou
    inter = np.sum(infer_map * gt_map)
    union = np.sum(gt_map) + np.sum(infer_map * (gt_map == 0))
    ciou = inter / union

    return ciou, inter, union


if __name__ == '__main__':
    main()
