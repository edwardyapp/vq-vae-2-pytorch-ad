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


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)
        return torch.from_numpy(ar).long()


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


def validate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_accuracy = 0.0
    num_samples = 0

    with torch.no_grad():
        for top, bottom, label in loader:
            top = top.to(device)

            if args.hier == 'top':
                target = top
                out, _ = model(top)
            elif args.hier == 'bottom':
                bottom = bottom.to(device)
                target = bottom
                out, _ = model(bottom, condition=top)

            loss = criterion(out, target)
            total_loss += loss.item()

            _, pred = out.max(1)
            correct = (pred == target).float()
            total_accuracy += correct.sum().item()
            num_samples += target.numel()

    average_loss = total_loss / len(loader)
    accuracy = total_accuracy / num_samples

    return average_loss, accuracy


if __name__ == '__main__':
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

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=4, drop_last=False
    )

    ckpt = {}
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        epoch_str = os.path.splitext(args.ckpt)[0].split("_")[2]
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

    best_validation_loss = float('inf')
    patience = 20  # Number of epochs to wait before early stopping
    best_epoch = 0
    last_best_checkpoint_path = None

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    object_name = args.path.split('/')[-1]

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(train_loader) * args.epoch, momentum=None
        )

    for i in range(start_epoch, args.epoch):
        train(args, i, train_loader, model, optimizer, scheduler, device)

        if i % 5 == 0:  # Validate every 5 epochs (adjust as needed)
            validation_loss, validation_accuracy = validate(model, val_loader, device)

            print(
                f"Epoch {i + 1}, Validation Loss: {validation_loss:.5f}, Validation Accuracy: {validation_accuracy:.5f}")

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                # Remove the old checkpoint if it exists
                if last_best_checkpoint_path:
                    os.remove(last_best_checkpoint_path)

                # Update the checkpoint filename to include loss and accuracy
                checkpoint_filename = f'{object_name}_pixelsnail_{args.hier}_{str(i + 1).zfill(3)}_Loss{validation_loss:.5f}_Acc{validation_accuracy:.5f}.pt'
                checkpoint_path = os.path.join('checkpoint', object_name, checkpoint_filename)

                if not os.path.exists(os.path.join('checkpoint', object_name)):
                    os.makedirs(os.path.join('checkpoint', object_name))

                torch.save(
                    {'model': model.module.state_dict(), 'args': args},
                    checkpoint_path,
                )

                # Update the last_best_checkpoint_path variable
                last_best_checkpoint_path = checkpoint_path

                # Keep track of the best epoch for early stopping
                best_epoch = i
            else:
                # If validation loss doesn't improve, check if it's time to stop
                if i - best_epoch >= patience:
                    print("Early stopping. No improvement for {} epochs.".format(patience))
                    break