import time

import numpy as np
import torch
import torch.utils.data as data
from models.net import Net
from datasets.line_dataset import LineDataset


if __name__ == '__main__':
    device = torch.device("cuda")
    net = Net(2).to(device)
    indices = [14]
    test_set = LineDataset("./datasets/images", indices, 999999)
    test_dataloader = data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
    net.load_state_dict(torch.load("./result/2,3.ckpt", map_location=lambda storage, loc: storage))
    data_iter = iter(test_dataloader)
    n_index = 100
    deltas = []
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
        pred = net(img)
        left_delta = abs(label.data[0][0] - pred.data[0][0]).item() * 256
        right_delta = abs(label.data[0][1] - pred.data[0][1]).item() * 256
        if left_delta < 1.0:
            line_1px += 1
        if right_delta < 1.0:
            line_1px += 1
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        delta = abs(label.data[0][0] - pred.data[0][0]) + abs(label.data[0][1] - pred.data[0][1])
        deltas.append(int(delta.item() * 256))
        print("[%s] - [index: %6d]" % (t, i+1),
              "label:", (int(label.data[0][0].item() * 256), int(label.data[0][1].item() * 256)),
              "pred:", (int(pred.data[0][0].item() * 256), int(pred.data[0][1].item() * 256)),
              "delta(pixel):", int(delta.item() * 256))
    print("average: %.2f, p75: %d, p90: %d, p95: %d, p99: %d" % (
              np.average(deltas),
              np.percentile(deltas, 75, interpolation='lower'),
              np.percentile(deltas, 90, interpolation='lower'),
              np.percentile(deltas, 95, interpolation='lower'),
              np.percentile(deltas, 99, interpolation='lower')))
    print("line_delta < 1:", line_1px)

