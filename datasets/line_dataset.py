import random

import numpy
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os


class LineDataset(data.Dataset):
    def __init__(self, base_dir, indices, length, size=(256, 16)):
        super(LineDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        template = "虚%d.bmp"
        self.size = size
        self.length = length  # 动态生成长度
        self.imgs = []  # 原始图像
        labels = open(os.path.join(base_dir, "labels.txt"), 'r').read().split("\n")
        labels = {int(i.split(",")[0]): i for i in labels}
        for index in indices:
            filename = template % index
            path = os.path.join(base_dir, filename)
            img_fp = open(path, 'rb')
            img = Image.open(img_fp).convert("L")
            img_fp.close()
            label = [int(i) for i in labels[index].split(",")]
            item = (img, label[1], label[2])  # 原图，x1，x2
            self.imgs.append(item)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        raw_img, x1, x2 = random.choice(self.imgs)
        raw_width, raw_heignt = raw_img.size

        # # 方案一、模拟人工画bounding-box随机裁剪，包含x1 x2区域
        # padding_ration_x = 10  # 左右至少留白10%
        # padding_ration_y = 20  # 上下至少留白20%
        # x1_percent, x2_percent = int(x1*100 / raw_width), int(x2*100 / raw_width)
        # # 模拟画框，位置以百分比表示
        # left_percent = random.randint(padding_ration_x, x1_percent - padding_ration_x)
        # right_percent = random.randint(x2_percent + padding_ration_x, 100 - padding_ration_x)
        # top_percent = random.randint(0, 100 - 2*padding_ration_y)
        # bottom_percent = random.randint(top_percent+padding_ration_y, 100 - padding_ration_y)
        # # 百分比转像素坐标
        #
        # left_pixel = int(raw_width * left_percent / 100)
        # right_pixel = int(raw_width * right_percent / 100)
        # top_pixel = int(raw_heignt * top_percent / 100)
        # bottom_pixel = int(raw_heignt * bottom_percent / 100)

        # 方案二、取消resize，直接指定右下角点坐标
        left_pixel = random.randint(x2 - 256, x1)
        right_pixel = left_pixel + self.size[0]
        top_pixel = random.randint(0, raw_heignt - 20)
        bottom_pixel = top_pixel + self.size[1]

        # crop
        img = raw_img.crop((left_pixel, top_pixel, right_pixel, bottom_pixel))
        # 转换后的x1 x2
        x1 -= left_pixel
        x2 -= left_pixel
        # 归一化
        x1 /= img.size[0]
        x2 /= img.size[0]
        # resize
        img = img.resize(self.size)
        img = self.transform(img)
        return img, numpy.array([x1, x2], dtype=numpy.float32), (left_pixel, right_pixel)


# test
if __name__ == '__main__':
    dataset = LineDataset("./images", (2, 3, 4, 5, 6, 10), 100)
    img, label = dataset.__getitem__(1)
    print(img.size, label)
    img.show()
