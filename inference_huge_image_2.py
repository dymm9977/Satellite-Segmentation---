import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from catalyst.dl import SupervisedRunner
from skimage.morphology import remove_small_holes, remove_small_objects
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision_2 import *
import random
import os
from collections.abc import MutableMapping

# 设置随机种子，保证结果可复现
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 将UAVid数据集的掩码转换为RGB图像
# inference_huge_image_2.py
def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    # 根据PALETTE映射颜色
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]      # Background
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [144, 238, 144]    # Class1
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]    # Class2
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]  # Class3
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [139, 69, 19]   # Class4
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb

# 对图像进行填充，使其大小为补丁大小的整数倍
def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]
    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    h, w = oh + height_pad, ow + width_pad
    pad = albu.PadIfNeeded(min_height=h, min_width=w, border_mode=0,
                           position='bottom_right', value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


# 定义推理数据集
class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results
    def __len__(self):
        return len(self.tile_list)


# 为一张大图像创建数据集
def make_dataset_for_one_huge_image(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)
    output_height, output_width = image_pad.shape[0], image_pad.shape[1]
    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x + patch_size[0], y: y + patch_size[1]]
            tile_list.append(image_tile)
    dataset = InferenceDataset(tile_list=tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape


# 主函数
def main():
    # 直接在代码中设置参数
    args = argparse.Namespace(
        #image_path='C:/Users/public_account/PycharmProjects/PythonProject/GeoSeg/data/mydataset/test',
        config_path=Path('C:/Users/public_account/PycharmProjects/PythonProject/GeoSeg/mydataset_unetformer_2.py'),
        output_path=Path('C:/Users/public_account/PycharmProjects/PythonProject/GeoSeg/fig_results/mydataset/unetformer_huge'),
        tta='lr',
        patch_height=256,
        patch_width=256,
        batch_size=2,
        dataset='uavid'
    )

    seed_everything(42)
    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    device = torch.device(f'cuda:{int(config.gpus[0])}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    '''elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
'''

    img_paths = ["C:/Users/public_account/Desktop/satellite_segment/3.png"]  # 示例测试图像路径
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape = \
            make_dataset_for_one_huge_image(img_path, patch_size)
        output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
        output_tiles = []
        k = 0
        with torch.no_grad():
            dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                    drop_last=False, shuffle=False)
            for input in tqdm(dataloader):
                raw_predictions = model(input['img'].cuda(config.gpus[0]))
                raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                predictions = raw_predictions.argmax(dim=1)
                image_ids = input['img_id']
                for i in range(predictions.shape[0]):
                    raw_mask = predictions[i].cpu().numpy()
                    mask = raw_mask
                    output_tiles.append((mask, image_ids[i].cpu().numpy()))

        for m in range(0, output_height, patch_size[0]):
            for n in range(0, output_width, patch_size[1]):
                output_mask[m:m + patch_size[0], n: n + patch_size[1]] = output_tiles[k][0]
                k = k + 1

        output_mask = output_mask[-img_shape[0]:, -img_shape[1]:]

        if args.dataset == 'uavid':
            output_mask = uavid2rgb(output_mask)
        else:
            output_mask = output_mask
        assert img_shape == output_mask.shape
        cv2.imwrite(os.path.join(args.output_path, img_name), output_mask)


if __name__ == "__main__":
    main()