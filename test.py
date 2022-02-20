import time

import numpy as np
import torch
import torch.utils.data as data
from models.net import Net
from datasets.line_dataset import LineDataset
from PIL import Image

index = 14


def count_top_n(items, n):
    items, items_count = np.unique(items, return_counts=True)
    items_count_sort = np.argsort(items_count)
    return np.asarray([items[items_count_sort[i]] for i in range(n)])


#  3-sigma 过滤太离谱的预测
def filter_n_sigma_lower_upper(items, n):
    mean = np.mean(items)
    std = np.std(items)
    lower = mean - n*std
    upper = mean + n*std
    return np.asarray([i for i in items if i>lower and i<upper])


if __name__ == '__main__':
    device = torch.device("cpu")
    net = Net(2).to(device)
    indices = [index]
    test_set = LineDataset("./datasets/images", indices, 999999)
    test_dataloader = data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
    net.load_state_dict(torch.load("./result/14.ckpt", map_location=lambda storage, loc: storage))
    data_iter = iter(test_dataloader)
    n_index = 100
    deltas = []
    preds_left = []
    preds_right = []
    line_1px = 0
    pic_1px = 0
    for i in range(n_index):
        try:
            batch_data = data_iter.next()
        except StopIteration as e:
            del data_iter
            data_iter = iter(test_dataloader)
            batch_data = data_iter.next()
        img = batch_data[0].to(device)
        label = batch_data[1].to(device)
        left, right = batch_data[2]
        pred = net(img)
        left_delta = abs(label.data[0][0] - pred.data[0][0]).item() * 256
        right_delta = abs(label.data[0][1] - pred.data[0][1]).item() * 256
        if left_delta < 1.0:
            line_1px += 1
        if right_delta < 1.0:
            line_1px += 1
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        preds_left.append(left + pred.data[0][0].item() * 256)
        preds_right.append(left + pred.data[0][1].item() * 256)
        delta = abs(label.data[0][0] - pred.data[0][0]) + abs(label.data[0][1] - pred.data[0][1])
        deltas.append(int(delta.item() * 256))
        print("[%s] - [index: %6d]" % (t, i+1),
              "label:", (int(label.data[0][0].item() * 256), int(label.data[0][1].item() * 256)),
              "pred:", (int(pred.data[0][0].item() * 256), int(pred.data[0][1].item() * 256)),
              "left, right", (left, right),
              "delta(pixel):", int(delta.item() * 256))
    print("average: %.2f, p75: %d, p90: %d, p95: %d, p99: %d" % (
              np.average(deltas),
              np.percentile(deltas, 75, interpolation='lower'),
              np.percentile(deltas, 90, interpolation='lower'),
              np.percentile(deltas, 95, interpolation='lower'),
              np.percentile(deltas, 99, interpolation='lower')))
    print("line_delta < 1:", line_1px)

    left_avg = np.average(preds_left)
    right_avg = np.average(preds_right)

    # 3-sigma
    left_filterd = filter_n_sigma_lower_upper(preds_left, 1)
    right_filterd = filter_n_sigma_lower_upper(preds_right, 1)

    label_file = open("./datasets/images/labels.txt", "r")
    labels = label_file.read().split("\n")
    label_file.close()
    label = ()
    for label_row in labels:
        label_split = label_row.split(",")
        if int(label_split[0]) == index:
            label = (int(label_split[1]), int(label_split[2]))

    print(label, (left_avg, right_avg), (np.average(left_filterd), np.average(right_filterd)))
    pred = (int(np.around(left_avg)), int(np.around(right_avg)))
    print(index, label, pred)

    img_fp = open("./datasets/images/虚%d.bmp"%index, "rb")
    img = Image.open(img_fp).convert("RGB")
    img_mat = np.array(img)
    print(img_mat.shape)
    print(img_mat[:,0,:].shape)
    img_mat[:,label[0],:] = [0, 255, 0]
    img_mat[:,label[1],:] = [0, 255, 0]
    img_mat[:,pred[0],:] = [255, 0, 0]
    img_mat[:,pred[1],:] = [255, 0, 0]
    img = Image.fromarray(img_mat)
    img.save("./result/%d.bmp"%index)


