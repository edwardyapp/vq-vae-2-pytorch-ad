import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE

import os
import sys
from torch import nn
import distributed as dist

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict

def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, _, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.add_argument("--resize", type=int, default=292)
    parser.add_argument("--centerCrop", type=int, default=256)
    parser.add_argument("--randomCrop", type=int, default=282)
    parser.add_argument("--randomRotation", type=int, default=2)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('--batch', type=int, default=128)

    args = parser.parse_args()

    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=(-args.randomRotation, args.randomRotation)),
            transforms.Resize(args.resize),
            transforms.RandomCrop(args.randomCrop),
            transforms.CenterCrop(args.centerCrop),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFileDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    model = VQVAE()

    # Special treatment if multiple GPUs were used to train VQ-VAE
    if args.gpu:
        state_dict = torch.load(args.ckpt)

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        # load params
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(args.ckpt))

    model = model.to(device)

    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    if not os.path.exists(args.name):
        os.makedirs(args.name)

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)