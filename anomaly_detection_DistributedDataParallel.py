# Import necessary libraries and modules
import os
import argparse
import torch
import torch.nn.functional as F
import mvtec as mvtec
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from torchvision.utils import save_image
from pixelsnail import PixelSNAIL
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE
from skimage import morphology
from skimage.segmentation import mark_boundaries
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch.utils.data.distributed import DistributedSampler

# Function to plot and save the anomaly detection results
def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()

# Function to denormalize the image
# https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x

# Function to load dataset folder
def load_dataset_folder(self):
    phase = 'train' if self.is_train else 'test'
    x, y, mask = [], [], []

    img_dir = os.path.join(self.dataset_path, self.class_name, phase)
    gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

    img_types = sorted(os.listdir(img_dir))
    for img_type in img_types:

        # load images
        img_type_dir = os.path.join(img_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                 for f in os.listdir(img_type_dir)
                                 # if f.endswith('.png')])
                                 if f.endswith('.jpg')])
        x.extend(img_fpath_list)

        # load gt labels
        if img_type == 'good':
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
        else:
            y.extend([1] * len(img_fpath_list))
            gt_type_dir = os.path.join(gt_dir, img_type)
            img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
            gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.jpg')  # '_mask.png')
                             for img_fname in img_fname_list]
            mask.extend(gt_fpath_list)

    assert len(x) == len(y), 'number of x and y should be same'

    return list(x), list(y), list(mask)

# Function to sample the model without gradients
@torch.no_grad()
def sample_model(model, device, batch, size, temperature, codes, condition=None):
    row = torch.clone(codes)
    cache = {}
    loss = torch.tensor([]).to(device)

    # if condition is None:
    #     threshold = 7
    # else:
    #     threshold = 9

    threshold = 0.0005

    countUpdates = 0

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            loss_temp = F.cross_entropy(out[:, :, i, j], row[:, i, j], reduction='none')
            loss = torch.cat((loss_temp, loss))

            # The probability distribution for the current component (i, j) is computed using softmax
            # This prob variable holds the probability distribution of the current component conditioned only on the
            # previous sequences (i.e., the components that come before the current component in the raster order).
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)

            sample = torch.multinomial(prob, 1).squeeze(-1)

            # if torch.any(loss_temp > threshold):
            #     countUpdates += 1

            # Check for the 8-neighbors
            neighbors = []
            for dx, dy in [(0, 1), (1, -1), (1, 0), (1, 1)]:
                x, y = i + dx, j + dy
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    neighbors.append((x, y))

            current_prob = [prob[b, row[b, i, j].item()] for b in range(len(sample))]

            # Convert current_prob to a tensor
            current_prob_tensor = torch.tensor(current_prob)

            # Calculate the minimum conditional probability for the bottom half of the 8-neighbors
            # In the very last row and column, there are no neighbors, hence the check to make sure that the  neighbors
            # list is not empty.
            min_neighbor_prob = [min([prob[b, row[b, x, y].item()] for (x, y) in neighbors]) for b in range(len(sample))
                                 if neighbors]
            min_neighbor_prob = [min([prob[b, row[b, x, y].item()] for (x, y) in neighbors]) if neighbors else 0 for b
                                 in range(len(sample))]

            # Convert min_neighbor_prob to a tensor
            min_neighbor_prob_tensor = torch.tensor(min_neighbor_prob)

            # Perform resampling if the current component and at least one neighbor have a likelihood less than the threshold
            update_condition = (current_prob_tensor < threshold) & (min_neighbor_prob_tensor < threshold)

            # if torch.any(update_condition): print("update_condition: ", update_condition)

            row[update_condition, i, j] = sample[update_condition]

    # plt.hist(loss.cpu().numpy())
    # plt.savefig('foo.png', bbox_inches='tight')
    # plt.show()

    # print("np.percentile(loss.cpu().numpy(), 90): ", np.percentile(loss.cpu().numpy(), 90))
    # print("1.5IQR: ", 1.5 * (np.percentile(loss.cpu().numpy(), 75) - np.percentile(loss.cpu().numpy(), 25)))
    # print("countUpdates: ", countUpdates)

    return row

# Function to extract features from the model and calculate metrics
def extract(loader, model_vqvae, device, model_top, model_bottom, batch, size_top, size_bottom, temperature):

    counter = 0
    img_list = []
    gt_mask_list = []
    dist_list = []

    for (img, y, mask) in tqdm(loader):

        img = img.to(device)

        # List of test images (img_list) and ground truth masks (gt_mask_list) required for visualisation of results
        img_list.extend(img.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())

        # Need to call module first if using DataParallel
        # _, _, _, id_t, id_b = model_vqvae.encode(img)
        _, _, _, id_t, id_b = model_vqvae.module.encode(img)

        # Reconstructured images are an attempt to reproduce the input image whether good or bad
        # Need to call module first if using DataParallel
        # decoded_sample_reconstructed = model_vqvae.decode_code(id_t, id_b)
        decoded_sample_reconstructed = model_vqvae.module.decode_code(id_t, id_b)

        # Uncomment to view reconstructed image
        # decoded_sample_reconstructed = decoded_sample_reconstructed.clamp(-1, 1)
        # save_image(decoded_sample_reconstructed[0, :, :, :], f"reconstructed_{str(counter).zfill(5)}_{str(0).zfill(5)}.png", normalize=True, range=(-1, 1))

        # Given that reconstructed and restored images are in colour (three channels), they have to be first converted
        # to gray scale (single channel) in order to get the difference between the two images
        transform_grayscale = transforms.Grayscale()

        decoded_sample_reconstructed_gray = transform_grayscale(decoded_sample_reconstructed.to(device).float())

        # Modified top latent code
        top_sample = sample_model(model_top, device, batch, size_top, temperature, id_t)

        # Modified bottom latent code
        bottom_sample = sample_model(
            model_bottom, device, batch, size_bottom, temperature, id_b, condition=top_sample
        )

        # Restored image where anomalous regions are replaced by distribution corresponding to good parts
        decoded_sample_restored = model_vqvae.decode_code(top_sample, bottom_sample)

        decoded_sample_restored_gray = transform_grayscale(decoded_sample_restored.to(device).float())

        # L1 distance between each pixel for each image
        # dist_temp = torch.abs(decoded_sample_reconstructed_gray.squeeze(1) - decoded_sample_restored_gray.squeeze(1))

        # Calculate the L2 distance between the images for each image in the batch
        dist_temp = torch.sqrt((decoded_sample_reconstructed_gray.squeeze(1) -
                                          decoded_sample_restored_gray.squeeze(1)) ** 2)

        # List of pixel-level differences for each image required for visualisation of results
        dist_list.extend(dist_temp.cpu().detach().numpy().reshape(-1))

        # Uncomment to view restored image
        # decoded_sample = decoded_sample.clamp(-1, 1)
        # save_image(decoded_sample[0, :, :, :], f"restored_{str(counter).zfill(5)}_{str(0).zfill(5)}.png", normalize=True, range=(-1, 1))

    dist = np.asarray(dist_list)

    for i in range(dist.shape[0]):
        dist[i] = gaussian_filter(dist[i], sigma=4)

    # Normalization
    max_score = dist.max()
    min_score = dist.min()
    scores = (dist - min_score) / (max_score - min_score)

    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    total_pixel_roc_auc.append(per_pixel_rocauc)
    print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

    class_name = 'bottle'

    fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

    save_dir = args.save_path + '/' + f'pictures'
    os.makedirs(save_dir, exist_ok=True)
    plot_fig(img_list, scores, gt_mask_list, threshold, save_dir, class_name)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)

# Function to load the pre-trained model
def load_model(model, checkpoint, device, ngpu):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
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

    elif model == 'pixelsnail_bottom':
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
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Attempt to use multiple GPUS but code is slower
    # if ngpu > 1:
    #     model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))

    return model

def main_worker(gpu, ngpu, args):
    # Initialize the distributed environment
    torch.cuda.set_device(gpu)

    # Define the device variable
    device = torch.device(f"cuda:{gpu}")  # Add this line to define the device variable

    import sys

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )

    dist.init_process_group(backend='nccl', init_method=f"tcp://127.0.0.1:{port}", world_size=ngpu, rank=gpu)

    # Load models
    model_vqvae = load_model('vqvae', args.vqvae, device, ngpu)
    model_top = load_model('pixelsnail_top', args.top, device, ngpu)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device, ngpu)

    # Wrap models with DistributedDataParallel
    model_vqvae = torch.nn.parallel.DistributedDataParallel(model_vqvae, device_ids=[gpu])
    model_top = torch.nn.parallel.DistributedDataParallel(model_top, device_ids=[gpu])
    model_bottom = torch.nn.parallel.DistributedDataParallel(model_bottom, device_ids=[gpu])

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    test_dataset = mvtec.MVTecDataset(args.data_path, class_name="bottle", is_train=False)
    sampler = DistributedSampler(test_dataset)
    loader = DataLoader(test_dataset, batch_size=args.batch, sampler=sampler, num_workers=0, pin_memory=True)
    extract(loader, model_vqvae, device, model_top, model_bottom, args.batch, [32, 32], [64, 64], args.temp)

def main():
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('filename', type=str)

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('path', type=str)

    parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')

    args = parser.parse_args()

    ngpu = torch.cuda.device_count()
    if ngpu > 1:
        mp.spawn(main_worker, nprocs=ngpu, args=(ngpu,args))
    else:
        main_worker(0, ngpu, args)

# Main function to execute the anomaly detection pipeline
if __name__ == '__main__':
    main()