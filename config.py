import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--n_iters', type=int, default=100000)
train_arg.add_argument('--device', type=str, default="cuda")
train_arg.add_argument('--base_dir', type=str, default="./datasets/images")
train_arg.add_argument('--indices', type=str, default="2,3,4,5,6,10")
train_arg.add_argument('--batch_size', type=int, default=1)
train_arg.add_argument('--num_workers', type=int, default=1)
train_arg.add_argument('--optimizer', type=str, default='Adam')
train_arg.add_argument('--loss', type=str, default='MSE')
train_arg.add_argument('--lr', type=float, default=0.0001)

train_arg.add_argument('--info_interval', type=int, default=100)
train_arg.add_argument('--save_image_interval', type=int, default=100)
train_arg.add_argument('--save_model_interval', type=int, default=100)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed