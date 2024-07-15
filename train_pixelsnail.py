import argparse

import numpy as np
import torch
import os

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)

        if args.hier == 'top':
            target = top
            out, _ = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )

    return loss

class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()

def save_checkpoint(model, args, object_name, epoch, is_best):
    checkpoint_name = f'checkpoint/{object_name}/{object_name}_pixelsnail_{args.hier}_{str(epoch + 1).zfill(3)}.pt'
    torch.save(
        {'model': model.module.state_dict(), 'args': args},
        checkpoint_name,
    )
    if is_best:
        best_checkpoint_name = f'checkpoint/{object_name}/{object_name}_pixelsnail_{args.hier}_best.pt'
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            best_checkpoint_name,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=420)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        # Have to extract epoch before it is overwritten in "args = ckpt['args']"
        epoch_str = os.path.splitext(args.ckpt)[0].split("_")[-1]
        start_epoch = int(epoch_str)

        end_epoch = args.epoch

        args = ckpt['args']

        args.epoch = end_epoch
    else:
        start_epoch = 0

    if args.hier == 'top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    object_name = args.path.split('/')[-1]

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    best_loss = float('inf')

    for i in range(start_epoch, args.epoch):
        loss = train(args, i, loader, model, optimizer, scheduler, device)

        val_loss = loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, args, object_name, i, is_best=True)
        else:
            save_checkpoint(model, args, object_name, i, is_best=False)

        if i > 0:
            os.remove(f'checkpoint/{object_name}/{object_name}_pixelsnail_{args.hier}_{str(i).zfill(3)}.pt')

if __name__ == '__main__':
    main()