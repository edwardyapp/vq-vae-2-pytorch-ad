import argparse
import sys
import os
import torch
import distributed as dist
import shutil

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from vqvae import VQVAE
from scheduler import CycleScheduler

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

best_val_loss = float('inf')
epochs_without_improvement = 0

def train(epoch, train_loader, val_loader, model, optimizer, scheduler, device, class_name):
    global best_val_loss  # Declare best_val_loss as a global variable
    global epochs_without_improvement  # Declare epochs_without_improvement as a global variable

    if dist.is_primary():
        train_loader = tqdm(train_loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(train_loader):

        # model.eval()
        #
        # sample = img[:sample_size]
        #
        # with torch.no_grad():
        #     out, _ = model(sample)
        #
        # import numpy as np
        #
        # diff_img = np.abs(out - sample)
        #
        # threshold = 0.4
        #
        # threshold = (threshold * 0.5 + 0.5) * 255
        # diff_img[diff_img <= threshold] = 0
        #
        # anomaly_img = np.zeros_like(sample)
        # anomaly_img[:, :, :] = sample
        # anomaly_img[np.where(diff_img > 0)[0], np.where(diff_img > 0)[1]] = [200, 0, 0]
        # # anomaly_img[:, :, 0] = anomaly_img[:, :, 0] + 10.0 * np.mean(diff_img, axis=2)
        #
        # import matplotlib.pyplot as plt
        #
        # fig, plots = plt.subplots(1, 4)
        #
        # fig.set_figwidth(9)
        # fig.set_tight_layout(True)
        # plots = plots.reshape(-1)
        # plots[0].imshow(sample, label="real")
        # plots[1].imshow(out)
        # plots[2].imshow(diff_img)
        # plots[3].imshow(anomaly_img)
        #
        # plots[0].set_title("real")
        # plots[1].set_title("generated")
        # plots[2].set_title("difference")
        # plots[3].set_title("Anomaly Detection")
        # plt.show()

        ##############################################

        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            # val_loss = validate(model, validation_loader, device)

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     epochs_without_improvement = 0
            #     torch.save(model.state_dict(), f"checkpoint/{class_name}_best_model.pt")
            # else:
            #     epochs_without_improvement += 1

            # if args.early_stopping:
            #     if epochs_without_improvement >= args.patience:
            #         print(f"\nEarly stopping at epoch {epoch}")
            #         break
            #     else:
            #         print("\nepochs_without_improvement: ", epochs_without_improvement)

            train_loader.set_description(
                (
                    f"class_name: {class_name}; "
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}; "
                    # f"validation mse: {val_recon_loss.item():.5f}; "
                    # f"validation latent: {val_latent_loss.item():.3f}; validation avg mse: {val_mse:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                # sample = torch.unsqueeze(img[0], 0)
                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                # utils.save_image(
                #     torch.cat([out], 0),
                #     f"test/bad_predicted.png",
                #     nrow=1,
                #     normalize=True,
                #     range=(-1, 1),
                # )
                #
                # utils.save_image(
                #     torch.cat([sample], 0),
                #     f"test/bad_original.png",
                #     nrow=1,
                #     normalize=True,
                #     range=(-1, 1),
                # )
                #
                # import cv2
                # from skimage.metrics import structural_similarity
                # import numpy as np
                #
                # # Load images
                # before = cv2.imread('test/bad_predicted.png')
                # after = cv2.imread('test/bad_original.png')
                #
                # # Convert images to grayscale
                # before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
                # after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
                #
                # # Compute SSIM between the two images
                # (score, diff) = structural_similarity(before_gray, after_gray, full=True)
                # print("Image Similarity: {:.4f}%".format(score * 100))
                #
                # # The diff image contains the actual image differences between the two images
                # # and is represented as a floating point data type in the range [0,1]
                # # so we must convert the array to 8-bit unsigned integers in the range
                # # [0,255] before we can use it with OpenCV
                # diff = (diff * 255).astype("uint8")
                # diff_box = cv2.merge([diff, diff, diff])
                #
                # # Threshold the difference image, followed by finding contours to
                # # obtain the regions of the two input images that differ
                # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # contours = contours[0] if len(contours) == 2 else contours[1]
                #
                # mask = np.zeros(before.shape, dtype='uint8')
                # filled_after = after.copy()
                #
                # for c in contours:
                #     area = cv2.contourArea(c)
                #     if area > 40:
                #         x, y, w, h = cv2.boundingRect(c)
                #         cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
                #         cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
                #         cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
                #         cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                #         cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
                #
                # cv2.imshow('before', before)
                # cv2.imshow('after', after)
                # cv2.imshow('diff', diff)
                # cv2.imshow('diff_box', diff_box)
                # cv2.imshow('mask', mask)
                # cv2.imshow('filled after', filled_after)
                # cv2.waitKey()
                #
                # # diff_img = torch.abs(out - sample)
                #
                # diff_img = torch.abs(out - sample)
                #
                # threshold = 0.6
                #
                # # threshold = (threshold * 0.5 + 0.5) * 255
                # diff_img[diff_img <= threshold] = 0

                # anomaly_img = torch.zeros_like(sample)
                # anomaly_img[:, :, :] = sample
                # anomaly_img[torch.where(diff_img > 0)[0], torch.where(diff_img > 0)[1]] = [200/255, 0, 0]
                # anomaly_img[diff_img > 0][0] = 0
                # anomaly_img[:, :, 0] = anomaly_img[:, :, 0] + 10.0 * np.mean(diff_img, axis=2)

                # import matplotlib.pyplot as plt
                #
                # fig, plots = plt.subplots(1, 4)
                #
                # fig.set_figwidth(9)
                # fig.set_tight_layout(True)
                # plots = plots.reshape(-1)
                #
                # plots[0].imshow(torch.squeeze(sample.cpu()).permute(1, 2, 0), label="real")
                # plots[1].imshow(torch.squeeze(out.cpu()).permute(1, 2, 0))
                # plots[2].imshow(torch.squeeze(diff_img.cpu()).permute(1, 2, 0))
                # plots[3].imshow(torch.squeeze(anomaly_img.cpu()).permute(1, 2, 0))
                #
                # plots[0].set_title("real")
                # plots[1].set_title("generated")
                # plots[2].set_title("difference")
                # plots[3].set_title("Anomaly Detection")
                # plt.show()

                sample_folder = "sample"
                if not os.path.exists(sample_folder):
                    os.makedirs(sample_folder)

                # previous_sample_path = f"sample/{str(epoch).zfill(5)}_{str(i).zfill(5)}.png",
                # if os.path.exists(previous_sample_path[0]):
                #     os.remove(previous_sample_path[0])

                sample_class_name_folder = "sample/" + class_name
                if not os.path.exists(sample_class_name_folder):
                    os.makedirs(sample_class_name_folder)

                previous_image_path = f"sample/" + class_name + '/' + f"{class_name}_{str(epoch).zfill(5)}_{str(i).zfill(5)}.png"
                if os.path.exists(previous_image_path):
                    os.remove(previous_image_path)
                utils.save_image(
                    torch.cat([sample, out], 0),
                    "sample/" + class_name + '/' + f"{class_name}_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()

    val_mse_sum = 0
    val_mse_n = 0

    if dist.is_primary():
        val_loader = tqdm(val_loader)

    for i, (img, label) in enumerate(val_loader):
        model.zero_grad()
        img = img.to(device)
        out, val_latent_loss = model(img)
        val_recon_loss = criterion(out, img)
        val_latent_loss = val_latent_loss.mean()

        val_part_mse_sum = val_recon_loss.item() * img.shape[0]
        val_part_mse_n = img.shape[0]
        comm = {"mse_sum": val_part_mse_sum, "mse_n": val_part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            val_mse_sum += part["mse_sum"]
            val_mse_n += part["mse_n"]

        if dist.is_primary():
            val_loader.set_description(
                (
                    f"class_name: {class_name}; "
                    f"epoch: {epoch + 1}; "
                    f"validation mse: {val_recon_loss.item():.5f}; "
                    f"validation latent: {val_latent_loss.item():.3f}; validation avg mse: {val_mse_sum / val_mse_n:.5f}; "
                )
            )

def validate(model, val_loader, device):
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for img, _ in val_loader:
            img = img.to(device)
            out, latent_loss = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            total_loss += recon_loss.item() * img.shape[0]
            total_samples += img.shape[0]

    return recon_loss, latent_loss, total_loss / total_samples

def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

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

    # # Determine the available GPU memory size
    # gpu_memory_size = torch.cuda.get_device_properties(device).total_memory
    #
    # # Calculate the maximum power of 2 that can fit within the available memory
    # max_power_of_2 = int(gpu_memory_size / (1024 * 1024 * 1024) * 0.8)  # Use 80% of the available memory
    # batch_size = 1
    # while batch_size * 2 <= max_power_of_2:
    #     batch_size *= 2

    dataset = datasets.ImageFolder(args.path, transform=transform)
    # sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    from torch.utils.data import random_split

    # Assuming 'dataset' is your training dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = dist.data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    val_sampler = dist.data_sampler(val_dataset, shuffle=True, distributed=args.distributed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch // args.n_gpu, sampler=train_sampler, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=args.batch // args.n_gpu, sampler=val_sampler, num_workers=0)

    model = VQVAE()

    model = model.to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
            find_unused_parameters=True,
        )

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(train_loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    import os

    if args.ckpt is not None:
        epoch_str = os.path.splitext(args.ckpt)[0].split("_")[-1]
        start_epoch = int(epoch_str)
    else:
        start_epoch = 0

    # Remove the trailing slash, if present
    if args.path.endswith("/"):
        class_name = args.path.split("/")[-3]
    else:
        class_name = args.path.split("/")[-2]

    for i in range(start_epoch, args.epoch):
        train(i, train_loader, val_loader, model, optimizer, scheduler, device, class_name)

        if dist.is_primary():
            if args.early_stopping and epochs_without_improvement >= args.patience:
                break  # Stop training loop

            previous_checkpoint_path = f"checkpoint/{class_name}_vqvae_{str(i).zfill(3)}.pt"
            if os.path.exists(previous_checkpoint_path):
                os.remove(previous_checkpoint_path)
            torch.save(model.state_dict(), f"checkpoint/{class_name}_vqvae_{str(i + 1).zfill(3)}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--resize", type=int, default=292)
    parser.add_argument("--centerCrop", type=int, default=256)
    parser.add_argument("--randomCrop", type=int, default=282)
    parser.add_argument("--randomRotation", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)  # Number of epochs to wait for improvement before stopping
    parser.add_argument("--batch", type=int, default=256)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
