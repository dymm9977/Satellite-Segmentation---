import os
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# 设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_size = 256  # 每个小块的大小
stride = 128  # 重叠部分的步长

# 数据加载和预处理
def load_img(path, grayscale=False):
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = np.array(img, dtype="float32")   # 转换为 float32 并归一化
    return img

# SegNet 模型定义
class SegNet(nn.Module):
    def __init__(self, n_classes):
        super(SegNet, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 预测函数
def predict(model_path, test_set):
    model = SegNet(n_classes=5)  # n_classes 确保与训练时相同
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    transform = transforms.Compose([transforms.ToTensor()])

    for n, path in enumerate(test_set):
        image = load_img(path)
        h, w, _ = image.shape
        padding_h = (h // stride + 1) * stride  # 填充，使得图像可以被 stride 整除
        padding_w = (w // stride + 1) * stride

        # 对图像进行填充，以适应步长
        padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.float32)
        padding_img[:h, :w, :] = image

        # 转换为 PyTorch tensor
        padding_img = np.transpose(padding_img, (2, 0, 1))  # 从 HWC 转换为 CHW 格式
        padding_img = torch.from_numpy(padding_img).unsqueeze(0)  # 添加批次维度

        if torch.cuda.is_available():
            padding_img = padding_img.cuda()

        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)

        # 使用滑动窗口方式处理填充后的图像
        for i in range(0, padding_h - img_size + 1, stride):
            for j in range(0, padding_w - img_size + 1, stride):
                # 截取小块
                patch = padding_img[:, :, i:i + img_size, j:j + img_size]

                with torch.no_grad():
                    output = model(patch)
                    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                # 将预测结果放回到大图的对应位置
                mask_whole[i:i + img_size, j:j + img_size] = pred

        # 去掉填充的部分，恢复为原图大小
        mask_whole = mask_whole[:h, :w]

        # 转换为图像并保存
        mask_image = Image.fromarray(mask_whole.astype(np.uint8))
        mask_image.save(f'./predict/pre_{n + 1}.png')

if __name__ == '__main__':
    TEST_SET = ["C:/Users/public_account/Desktop/1.png"]  # 示例测试图像路径
    predict(model_path='./segnet.model', test_set=TEST_SET)