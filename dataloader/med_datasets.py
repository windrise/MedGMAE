import os
import torch
from monai.data.dataset import Dataset
from monai import transforms
import random
import numpy as np


class LargeMedicalDataSets(Dataset):
    def __init__(self, args, is_val=False):

        self.sample_list = []
        ls = []

        for op in os.listdir(args.base_dir1):
            name = op.split("_")
            name = name[0] + "_" + name[1]
            # if name not in ["Dataset024_WORD", "Dataset009_AMOS",
            #                 "Dataset023_KiTS2023", "Dataset012_Lung", "Dataset013_HepaticVessel",
            #                 "Dataset014_Spleen", "Dataset015_Colon", "Dataset016_Pancreas", "Dataset017_Liver"]:
            #     self.sample_list.append(os.path.join(args.base_dir1, op))
            #     ls.append(name)
            #"Dataset060_TotalSegmentator",
            if name in ["Dataset100_AbdomenAtlasMini"]:
            # if name in ["Dataset060_TotalSegmentator"]:
                self.sample_list.append(os.path.join(args.base_dir1, op))
                ls.append(name)
        # ls = ls[:500]
        # self.sample_list = self.sample_list[:500]
        #for private in os.listdir(args.base_dir2):
        #    self.sample_list.append(os.path.join(args.base_dir2, private))
        #for private in os.listdir(args.base_dir3):
        #    self.sample_list.append(os.path.join(args.base_dir3, private))
        #print(len(args.base_dir4))
        #for private in os.listdir(args.base_dir4):
        #    self.sample_list.append(os.path.join(args.base_dir4, private))
        
            
        arr = np.array(ls)
        v, c = np.unique(arr, return_counts=True)
        for v, c in zip(v, c):
            print(v, c)
        # Shuffle the list in place
        random.shuffle(self.sample_list)

        if is_val:
            self.sample_list = self.sample_list[:min(5, len(self.sample_list))]
            self.transform = transforms.Compose([
                transforms.CenterSpatialCropd(
                    keys=["image"],
                    roi_size=(96, 96, 96)
                ),
                # transforms.ScaleIntensityRanged(
                #     keys=["image"],
                #     a_min=-175,
                #     a_max=250,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=True,
                # ),
                transforms.ToTensord(keys=["image"]),
            ])
            print("Total: {} validation 3d volumes for visualization".format(len(self.sample_list)))
        else:
            self.sample_list = self.sample_list[:int(len(self.sample_list) * args.data_ratio)]
            self.transform = transforms.Compose([
                # transforms.CenterSpatialCropd(
                #     keys=["image"],
                #     roi_size=(96, 96, 96)
                # ),
                transforms.RandCropByPosNegLabeld(
                    spatial_size=(96, 96, 96),
                    keys=["image"],
                    label_key="image",
                    # pos=3,
                    # neg=1,
                    # num_samples=4,
                    pos=3,
                    # neg=2,
                    num_samples=3,
                ),
                # transforms.ScaleIntensityRanged(
                #     keys=["image"],
                #     a_min=-175,
                #     a_max=250,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=True,
                # ),
                transforms.ToTensord(keys=["image"]),
            ])
            # self.transform = transforms.Compose([
            #     transforms.RandCropByPosNegLabeld(
            #         spatial_size=(96, 96, 96),
            #         keys=["image"],
            #         label_key="image",
            #         # pos=3,
            #         # neg=1,
            #         # num_samples=4,
            #         pos=6,
            #         neg=2,
            #         num_samples=8,
            #     ),
            #     transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            #     transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            #     transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            #     transforms.RandRotate90d(keys=["image"], prob=0.3, max_k=3),
            #     # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            #     # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            #     transforms.ToTensord(keys=["image"]),
            # ])
            print("Total: {} pretrained 3d volumes".format(len(self.sample_list)))
        
        self.total = len(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = torch.load(self.sample_list[idx])
        sample = self.transform({"image": case["image"]}) 

        return sample


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="dense_mae_vit_large_12p", choices=["vit_base", "vit_large", "vit_base_8p", "vit_large_12p", "dense_mae_vit_large_12p"], help='model')
    parser.add_argument('--base_dir1', type=str, default="/storage-6/fenghetang/open/open_source", help='dir')
    parser.add_argument('--base_dir4', type=str, default="/storage-6/fenghetang/pretrain_dataset/open_single/BMICV_CACHE", help='dir')
    parser.add_argument('--base_dir3', type=str, default="/storage-6/fenghetang/pretrain_dataset/open_single/CT_RATE_CACHE", help='dir')
    parser.add_argument('--base_dir2', type=str, default="/storage-6/fenghetang/pretrain_dataset/private_pair/pair", help='dir')
    parser.add_argument('--pretrained_ckpt', type=str, default="dense_mae_vit_large_12p_mix_ckpt_0200.pth", help='dir')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
    parser.add_argument('--epochs', type=int, default=100, help='train epoch')
    parser.add_argument('--img_size', type=int, default=96, help='img size of per batch')
    parser.add_argument('--patch_size', type=int, default=12, help='patch size of per img')
    parser.add_argument('--in_chans', type=int, default=1, help='input channels')
    parser.add_argument('--pos_embed_type', type=str, default="sincos")
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--ckpt_dir', type=str, default="./ckpt/dense_mix_p12_50k")
    parser.add_argument('--save_ckpt_dir', type=str, default="./ckpt/dense_mix_p12_50k")
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--wde', type=float, default=0.2)
    parser.add_argument('--wp_ep', type=int, default=5)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    args = parser.parse_args()
    LargeMedicalDataSets(args=args)
