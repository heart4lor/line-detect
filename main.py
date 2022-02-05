import time
import torch
import torch.nn as nn
import torch.utils.data as data
from models.net import Net
from datasets.line_dataset import LineDataset
import torch.optim as optim
from itertools import chain
from config import get_config


if __name__ == '__main__':
    config, unparsed = get_config()
    device = torch.device(config.device)
    net = Net(2).to(device)
    indices = [int(i) for i in config.indices.split(",")]
    train_set = LineDataset(config.base_dir, indices, 999999)
    train_dataloader = data.DataLoader(
        dataset=train_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    loss_dict = {  # 只取了适用回归的loss
        "MSE": nn.MSELoss().to(device),
        "L1": nn.L1Loss().to(device),
        "SmoothL1": nn.SmoothL1Loss().to(device)
    }
    optimizer_dict = {
        "Adam": optim.Adam(chain(net.parameters()), lr=config.lr, betas=(0.9, 0.999))
    }
    criterion = loss_dict[config.loss]
    optimizer = optimizer_dict[config.optimizer]
    data_iter = iter(train_dataloader)
    n_iters = config.n_iters
    for cur_iter in range(1, n_iters+1):
        try:
            batch_data = data_iter.next()
        except StopIteration as e:
            del data_iter
            data_iter = iter(train_dataloader)
            batch_data = data_iter.next()
        img = batch_data[0].to(device)
        label = batch_data[1].to(device)
        net.zero_grad()
        pred = net(img)
        loss = criterion(pred, label)
        if cur_iter % config.info_interval == 0:
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            delta = abs(label.data[0][0] - pred.data[0][0]) + abs(label.data[0][1] - pred.data[0][1])
            print("[%s] - [iter: %6d]" % (t, cur_iter),
                  "loss: %.8f" % loss.item(),
                  "label:", (int(label.data[0][0].item()*256), int(label.data[0][1].item()*256)),
                  "pred:", (int(pred.data[0][0].item()*256), int(pred.data[0][1].item()*256)),
                  "delta(pixel):", int(delta.item()*256))
        if cur_iter % config.save_model_interval == 0:
            torch.save(net.state_dict(), config.save_model_path)
        loss.backward()
        optimizer.step()
