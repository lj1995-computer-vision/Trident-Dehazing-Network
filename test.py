import argparse
import torch
import time,os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from TDN import Net

parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--checkpoint", default="TDN_NTIRE2020_Dehazing.pt", type=str, help="model path")
parser.add_argument("--inp", default="input",type=str, help="test images path")
parser.add_argument("--opt", default="output",type=str, help="output images path")

os.environ["CUDA_VISIBLE_DEVICES"]='1'

opt = parser.parse_args()
print(opt)

net = Net(pretrained=False)
checkpoint=torch.load(opt.checkpoint)
net.load_state_dict(checkpoint)
net.eval()
net = nn.DataParallel(net, device_ids=[0]).cuda()

def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

images = [os.path.join(opt.inp, x) for x in os.listdir(opt.inp) if is_image_file(x)]
total_t=0

def forward_chop(*args, forward_function=None,shave=12, min_size=16000000):#160000
    # These codes are from https://github.com/thstkdgus35/EDSR-PyTorch
    # if self.input_large:
    #     scale = 1
    # else:
    #     scale = self.scale[self.idx_scale]
    scale = 1
    # n_GPUs = min(self.n_GPUs, 4)
    n_GPUs = 1
    _, _, h, w = args[0].size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    list_x = [[
        a[:, :, 0:h_size, 0:w_size],
        a[:, :, 0:h_size, (w - w_size):w],
        a[:, :, (h - h_size):h, 0:w_size],
        a[:, :, (h - h_size):h, (w - w_size):w]
    ] for a in args]

    list_y = []
    if w_size * h_size < min_size:
        for i in range(0, 4, n_GPUs):
            x = [torch.cat(_x[i:(i + n_GPUs)], dim=0) for _x in list_x]
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y):
                    _list_y.extend(_y.chunk(n_GPUs, dim=0))
    else:
        for p in zip(*list_x):
            y = forward_chop(*p, forward_function=forward_function,shave=shave, min_size=min_size)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    b, c, _, _ = list_y[0][0].size()
    y = [_y[0].new(b, c, h, w) for _y in list_y]
    for _list_y, _y in zip(list_y, y):
        _y[:, :, :h_half, :w_half] \
            = _list_y[0][:, :, :h_half, :w_half]
        _y[:, :, :h_half, w_half:] \
            = _list_y[1][:, :, :h_half, (w_size - w + w_half):]
        _y[:, :, h_half:, :w_half] \
            = _list_y[2][:, :, (h_size - h + h_half):, :w_half]
        _y[:, :, h_half:, w_half:] \
            = _list_y[3][:, :, (h_size - h + h_half):, (w_size - w + w_half):]

    if len(y) == 1: y = y[0]

    return y

def forward_x8(*args, forward_function=None):
    # These codes are from https://github.com/thstkdgus35/EDSR-PyTorch
    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()
        return ret

    list_x = []
    for a in args:
        x = [a]
        for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

        list_x.append(x)

    list_y = []
    for x in zip(*list_x):
        y = forward_function(*x)
        if not isinstance(y, list): y = [y]
        if not list_y:
            list_y = [[_y] for _y in y]
        else:
            for _list_y, _y in zip(list_y, y): _list_y.append(_y)

    for _list_y in list_y:
        for i in range(len(_list_y)):
            if i > 3:
                _list_y[i] = _transform(_list_y[i], 't')
            if i % 4 > 1:
                _list_y[i] = _transform(_list_y[i], 'h')
            if (i % 4) % 2 == 1:
                _list_y[i] = _transform(_list_y[i], 'v')

    y = [torch.cat(_y, dim=0).mean(dim=0) for _y in list_y]#, keepdim=True
    if len(y) == 1: y = y[0]

    return y

for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    im = Image.open(im_path)
    im1 = ToTensor()(im)
    im1 = Variable(im1).cuda().unsqueeze(0)
    t0=time.time()
    with torch.no_grad():
        im=forward_x8(im1,forward_function=net.forward)# self ensemble
        # im = net(im1).squeeze(0)# no self ensemble
        # im = forward_chop(im1, forward_function=net.forward).squeeze(0)# use it when graphics memory is not enough
    total_t=total_t+time.time()-t0
    im = im.cpu().data
    im = ToPILImage()(im)
    im.save('%s/%s' % (opt.opt,filename))

print(total_t)