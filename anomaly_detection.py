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
import cv2
import skimage.metrics as metrics
import time
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cm

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
from sklearn.metrics import confusion_matrix, f1_score
from tabulate import tabulate
from scipy.ndimage import zoom
from torch import nn

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict

# Generative capability based on training data or simply modify the relevant latent codes so that the assumed image of a
# defective part in testing looks like a normal part
generateImageFromScratch = False

# Bottom sample could be conditioned on either the original or modified (based on threshold) top latent code
conditionedOnModifiedTopSample = True

# If True, the number of times the pixelSnail output is obtained is equal to the latent code dimension, n; otherwise,
# it will n^2 times which will take much longer
quickRun = True

# According to https://doi.org/10.3390/app10238660, top of p. 12
probabilityThreshold = 0.00005

# According to https://doi.org/10.3390/app10238660, bottom of p. 8; the alternative being a Gaussian filter; anomalib
# has a far more complicated way of performing Gaussian filtering using kornia filters
bilateralFilter = False

# According to https://doi.org/10.3390/app10238660, bottom of p. 7; the alternative being updating based on the current
# latent code (i,j)
updateBasedOnNeighbours = True

# Different ways to calculate anomaly score based on l1 or l2 norm, RGB or grayscale image (maybe add in SSIM)
# Possible options: l1_RGB, l1_gray, l2_RGB, l2_gray
anomalyScore = 'l2_RGB'

# Options are "both", "topOnly" and "bottomOnly" that is to use both top and bottom priors or one or the other
prior = "both"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Function to plot and save the anomaly detection results
def plot_fig(test_img, scores, gts, pixel_threshold, save_dir, class_name, img_threshold_optimal, img_scores, gt_list):
    num = len(scores)
    vmax = scores.max()
    vmin = scores.min()
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    for i in range(num):
        img = test_img[i]
        img = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)  # denormalize
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i]
        normalized_heat_map = (heat_map - vmin) / (vmax - vmin)
        mask = scores[i]
        mask[mask > pixel_threshold] = 1
        mask[mask <= pixel_threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)

        # Additional information
        predicted_label = (img_scores[i] >= img_threshold_optimal).astype(int)
        true_label = gt_list[i]

        title = 'Sample {}\nImg_threshold: {:.3f}, Img_score: {:.3f}, Predicted: {}, True: {}, Pixel_threshold: {:.3f}'.format(
            i, img_threshold_optimal, img_scores[i], predicted_label, true_label, pixel_threshold)
        fig_img.suptitle(title)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(normalized_heat_map, cmap='jet', alpha=0.5, interpolation='none', vmin=0, vmax=1)
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

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'), cax=cbar_ax)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i) + '.pdf'), format='pdf', dpi=300, bbox_inches="tight")
        plt.close()

# Function to denormalize the image
# https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
def denormalization(x):
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
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

def write_statistics_to_file(save_dir, class_name, statistics_data, topOrBottom):
    file_path = os.path.join(save_dir, f'{class_name}_{topOrBottom}_latentCodeUpdateStatistics.txt')

    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, remove it
        os.remove(file_path)

    # Open the file in append mode and write the data
    with open(file_path, "a") as file:
        for metric, value in statistics_data:
            file.write(f"{metric}: {value}\n")

def append_update_metrics(test_metrics, metric_values, condition_name, count_updates, indices):
    metric_name = f"Number of updates for latent code (i, j), {condition_name} condition"
    test_metrics.append(metric_name)
    metric_values.append(sum([count_updates[i] for i in indices]))

def plot_histogram(selected_losses, class_name, save_dir, topOrBottom):
    plt.figure()

    for condition_name, selected_loss in zip(["Bad", "Good"], selected_losses):
        x_hist = torch.cat(selected_loss, dim=0).cpu()
        plt.hist(x_hist, bins=20, alpha=0.5, label=f'{condition_name} (max: %.2f)' % max(x_hist), log=True)

    plt.xlabel("Cross entropy loss (-)")
    plt.ylabel("Log frequency (-)")

    plt.legend()

    # Save the figure
    plt.savefig(f"{save_dir}/{class_name}_{topOrBottom}.pdf", bbox_inches='tight', dpi=300)
    plt.close()

def plot_and_write_statistics(losses, label, countUpdates, save_dir, class_name, topOrBottom):
    # Reshape losses
    reshaped_losses = [torch.stack(tensors).to(device) for tensors in losses]

    # Find indices where label is equal to 1
    indices_bad = torch.nonzero(label == 1).squeeze()
    indices_good = torch.nonzero(label == 0).squeeze()

    # Extract corresponding elements of reshaped_losses
    selected_losses_bad = [reshaped_losses[i] for i in indices_bad]
    selected_losses_good = [reshaped_losses[i] for i in indices_good]

    # Plot histograms for "Bad" and "Good" conditions in the same plot
    plot_histogram([selected_losses_bad, selected_losses_good], class_name, save_dir, topOrBottom)

    test_metrics = []
    metric_values = []

    # Append metrics for the "bad" condition
    append_update_metrics(test_metrics, metric_values, "bad", countUpdates, indices_bad)

    # Append metrics for the "good" condition
    append_update_metrics(test_metrics, metric_values, "good", countUpdates, indices_good)

    # Combine metrics and values into a list of lists
    statistics_data = list(zip(test_metrics, metric_values))

    # Usage in your code
    write_statistics_to_file(save_dir, class_name, statistics_data, topOrBottom)

def get_model_output(model, row, i, size, condition, cache):
    if updateBasedOnNeighbours:
        if i == size[0] - 1:
            return model(row[:, : i + 1, :], condition=condition, cache=cache)
        else:
            return model(row[:, : i + 2, :], condition=condition, cache=cache)
    else:
        return model(row[:, : i + 1, :], condition=condition, cache=cache)

def getLosses(size, model, row, condition, temperature, lossThreshold=1000, firstRound=None):
    losses = [[] for _ in range(len(row))]  # List of empty lists to store the current probabilities

    countUpdates = [0] * len(row)  # Initialize a counter for each image

    cache = {}

    for i in tqdm(range(size[0])):
    # for i in tqdm(range(2)): # For debugging purposes

        if firstRound or quickRun:
            out, cache = get_model_output(model, row, i, size, condition, cache)

        for j in range(size[1]):
        # for j in tqdm(range(2)): # For debugging purposes

            if (not quickRun) and (not firstRound):
                out, cache = get_model_output(model, row, i, size, condition, cache)

            # The probability distribution for the current component (i, j) is computed using softmax
            # This prob variable holds the probability distribution of the current component conditioned only on the
            # previous sequences (i.e., the components that come before the current component in the raster order).
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)

            # Picks the most probable sample but it may not correspond to maximum probability
            sample = torch.multinomial(prob, 1).squeeze(-1)

            # For use with loss-based threshold
            loss = F.cross_entropy(out[:, :, i, j], row[:, i, j], reduction='none')

            for idx, cp in enumerate(loss):
                losses[idx].append(cp)  # Append current_prob to the respective list

            if updateBasedOnNeighbours:
                # Check for the 8-neighbors
                neighbors = []
                for dx, dy in [(0, 1), (1, -1), (1, 0), (1, 1)]:
                    x, y = i + dx, j + dy
                    if 0 <= x < size[0] and 0 <= y < size[1]:
                        neighbors.append((x, y))

                # Calculate the maximum loss for the bottom half of the 8-neighbor
                # Arbitrarily large value of a 1000 so that it would exceed the threshold
                max_neighbor_loss_tensor = torch.max(torch.stack([F.cross_entropy(out[:, :, x, y], row[:, x, y],
                                                                                  reduction='none')
                                                                  for (x, y) in neighbors]), dim=0)[0] \
                    if neighbors else torch.full((len(sample),), 1000, device=device)

            if not firstRound:
                if updateBasedOnNeighbours:
                    update_condition = (loss > lossThreshold) & \
                                       (max_neighbor_loss_tensor > lossThreshold)
                else:
                    update_condition = (loss > lossThreshold)

                # Increment the counter for each image if update_condition is True for that image
                for idx, cond in enumerate(update_condition):
                    if cond:
                        countUpdates[idx] += 1

                row[update_condition, i, j] = sample[update_condition]

    return losses, row, countUpdates

# Function to sample the model without gradients
@torch.no_grad()
def sample_model(model, device, size, temperature, codes, condition=None, label=None, class_name=None, save_dir=None,
                 topOrBottom=None):

    if generateImageFromScratch:
        row = torch.zeros(len(codes), *size, dtype=torch.int64).to(device)
    else:
        row = torch.clone(codes)

    losses, _, _ = getLosses(size, model, row, condition, temperature, firstRound=True)

    lossThreshold = getLossThreshold(losses, label)

    losses, row, countUpdates = getLosses(size, model, row, condition, temperature,
                                          lossThreshold=lossThreshold, firstRound=False)

    plot_and_write_statistics(losses, label, countUpdates, save_dir, class_name, topOrBottom)

    return row

def getLossThreshold(losses, label):
    reshaped_losses = [torch.stack(tensors).to(device) for tensors in losses]
    indices = torch.nonzero(label == 0).squeeze()
    selected_losses = [reshaped_losses[i] for i in indices]
    x_hist_good = torch.cat(selected_losses, dim=0)

    lossThreshold = max(x_hist_good)
    lossThreshold = lossThreshold / 2

    return lossThreshold

# Function to extract features from the model and calculate metrics
def extract(loader, model_vqvae, device, model_top, model_bottom, batch, size_top, size_bottom, temperature, class_name):

    counter = 0
    img_list = []
    gt_list = []
    gt_mask_list = []
    dist_list = []
    total_roc_auc = []
    total_pixel_roc_auc = []

    save_dir = args.save_path + '/' + class_name
    os.makedirs(save_dir, exist_ok=True)

    inference_start = time.time()

    for (img, y, mask) in tqdm(loader):
        img = img.to(device)

        # List of test images (img_list) and ground truth masks (gt_mask_list) required for visualisation of results
        img_list.extend(img.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())

        _, _, _, id_t, id_b = model_vqvae.encode(img)

        # Reconstructured images are an attempt to reproduce the input image whether good or bad
        # Need to call module first if using DataParallel
        decoded_sample_reconstructed = model_vqvae.decode_code(id_t, id_b, prior)
        # decoded_sample_reconstructed = model_vqvae.module.decode_code(id_t, id_b)

        # Uncomment to view reconstructed image
        # decoded_sample_reconstructed = decoded_sample_reconstructed.clamp(-1, 1)
        # save_image(decoded_sample_reconstructed[0:9, :, :, :], f"reconstructed_{str(counter).zfill(5)}_{str(0).zfill(5)}.png", nrow=10, normalize=True, range=(-1, 1))

        # sample = img[:sample_size]
        #
        # # with torch.no_grad():
        # out, _ = model_vqvae(sample)
        #
        # save_image(
        #     torch.cat([sample, decoded_sample_reconstructed], 0),
        #     f"sample/{str(counter + 1).zfill(5)}_{str(0).zfill(5)}.png",
        #     nrow=sample_size,
        #     normalize=True,
        #     range=(-1, 1),
        # )

        # Given that reconstructed and restored images are in colour (three channels), they have to be first converted
        # to gray scale (single channel) in order to get the difference between the two images
        transform_grayscale = transforms.Grayscale()

        decoded_sample_reconstructed_gray = transform_grayscale(decoded_sample_reconstructed.to(device).float())

        img_gray = transform_grayscale(img.to(device).float())

        # Modified top latent code
        top_sample = sample_model(model_top, device, size_top, temperature, id_t, label=y, class_name=class_name,
                                  save_dir=save_dir, topOrBottom="top")

        # Specify the file path where you want to save the tensor
        file_path = save_dir + '/' + class_name + "_top_sample.pt"

        # Use torch.save to save the tensor to the specified file
        torch.save(top_sample, file_path)

        if not conditionedOnModifiedTopSample:
            top_sample = id_t

        bottom_sample = sample_model(model_bottom, device, size_bottom, temperature, id_b, condition=top_sample,
                                     label=y, class_name=class_name, save_dir=save_dir, topOrBottom="bottom")

        # Specify the file path where you want to save the tensor
        file_path = save_dir + '/' + class_name + "_bottom_sample.pt"

        # Use torch.save to save the tensor to the specified file
        torch.save(bottom_sample, file_path)

        # Restored image where anomalous regions are replaced by distribution corresponding to good parts
        decoded_sample_restored = model_vqvae.decode_code(top_sample, bottom_sample, prior)

        # decoded_sample_restored_gray = transform_grayscale(decoded_sample_restored.to(device).float())
        #
        # # L1 distance between each pixel for each image
        # dist_temp = torch.abs(decoded_sample_reconstructed_gray.squeeze(1) - decoded_sample_restored_gray.squeeze(1))

        # Calculate the L2 distance between the images for each image in the batch
        # Grayscale
        # dist_temp = torch.sqrt((decoded_sample_reconstructed_gray.squeeze(1) -
        #                                   decoded_sample_restored_gray.squeeze(1)) ** 2)

        # RGB
        if anomalyScore == 'l2_RGB':
            dist_temp = torch.sqrt(torch.sum(((decoded_sample_reconstructed - decoded_sample_restored) ** 2), dim=1))
            # dist_temp = torch.sqrt(torch.sum(((img - decoded_sample_restored) ** 2), dim=1))
            # dist_temp = torch.sum(((decoded_sample_reconstructed - decoded_sample_restored) ** 2), dim=1)
            # dist_temp = torch.mean(((decoded_sample_reconstructed - decoded_sample_restored) ** 2), dim=1)

        # Calculate the L1 distance between the images for each image in the batch
        # RGB
        # dist_temp = torch.sum(torch.abs(decoded_sample_reconstructed - img), dim=1)


        # Calculate the SSIM difference
        # Assuming decoded_sample_reconstructed and img are Torch tensors
        # decoded_sample_reconstructed_gray_np = decoded_sample_reconstructed_gray.detach().cpu().numpy()
        # img_gray_np = img_gray.detach().cpu().numpy()
        #
        # data_range = max(np.ptp(decoded_sample_reconstructed_gray_np), np.ptp(img_gray_np)) -\
        #              min(np.ptp(decoded_sample_reconstructed_gray_np), np.ptp(img_gray_np))
        #
        # dist_temp = 1 - metrics.structural_similarity(decoded_sample_reconstructed_gray_np.squeeze(), img_gray_np.squeeze(), win_size = 11, full=True, data_range=data_range)[1]
        # dist_temp = 1 - np.mean(dist_temp, axis=1)

        # List of pixel-level differences for each image required for visualisation of results
        if isinstance(dist_temp, np.ndarray):
            dist_list.extend(dist_temp)
        else:
            dist_list.extend(dist_temp.cpu().detach().numpy())

        # Uncomment to view restored image
        # decoded_sample = decoded_sample.clamp(-1, 1)
        # save_image(decoded_sample[0, :, :, :], f"restored_{str(counter).zfill(5)}_{str(0).zfill(5)}.png", normalize=True, range=(-1, 1))

    dist = np.asarray(dist_list)

    for i in range(dist.shape[0]):
        if bilateralFilter:
            dist[i] = cv2.bilateralFilter(dist[i], 20, 75, 75)
        else:
            dist[i] = gaussian_filter(dist[i], sigma=4)

    inference_time = (time.time() - inference_start) / len(test_dataset)
    # print('{} inference time per sample: {:.3f}'.format(class_name, inference_time))

    # Normalization
    max_score = dist.max()
    min_score = dist.min()
    scores = (dist - min_score) / (max_score - min_score)
    # print('max_score: ', max_score)
    # print('min_score: ', min_score)
    # print('scores size: ', scores.shape)

    # plot ROCAUC charts
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)  # max of all pixel scores
    # print('img_scores size: ', img_scores.shape)
    # print('number of img_thresholds if drop_intermediate = False: ', len(np.unique(img_scores)) + 1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, img_thresholds = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    total_roc_auc.append(img_roc_auc)
    fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
    # print('gt_list:', gt_list)
    # print('img_scores:', img_scores)
    # print('img_thresholds:', img_thresholds)
    # print('img_thresholds if drop_intermediate = False:', roc_curve(gt_list, img_scores, drop_intermediate=False)[2])

    # image threshold from roc curve
    distances = (tpr - 1.) ** 2 + fpr ** 2
    img_threshold = img_thresholds[np.argmin(distances)]

    # Pixel-level F1 score and pixel threshold
    gt_mask = np.asarray(gt_mask_list)
    gt_mask = (gt_mask > 0).astype(int)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    # print('thresholds: ', thresholds)
    # print('thresholds size: ', thresholds.shape)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    pixel_threshold = thresholds[np.argmax(f1)]
    pixel_F1Score = np.max(f1)

    # label, mask two types threshold
    # print('label based threshold: {:.3f}, pixel based threshold: {:.3f}'.format(img_threshold, threshold))

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    total_pixel_roc_auc.append(per_pixel_rocauc)
    fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % ("bottle", per_pixel_rocauc))
    # print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")
    # print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    # calculate image level F1 score and image threshold
    precision, recall, thresholds = precision_recall_curve(gt_list, img_scores)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    image_F1Score = np.max(f1)
    img_threshold_optimal = thresholds[np.argmax(f1)]

    # Sample test metrics and values (replace these with your actual data)
    test_metrics = ["image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score"]
    metric_values = [img_roc_auc, per_pixel_rocauc, image_F1Score, pixel_F1Score]

    # Combine metrics and values into a list of lists
    table_data = list(zip(test_metrics, metric_values))

    # Output as a formatted table
    formatted_table = tabulate(table_data, headers=["Test Metric", "Value"], tablefmt="pretty")
    print(formatted_table)

    # Specify the file path
    file_path = save_dir + '/' + class_name + "_testMetrics.txt"

    # Open the file in write mode and write the formatted table
    with open(file_path, "w") as file:
        file.write(formatted_table)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, '%s_lst_roc_curve.png' % "bottle"), dpi=100)

    # Calculate confusion matrix
    predicted_labels = (img_scores >= img_threshold_optimal).astype(int)
    confusion_mat = confusion_matrix(gt_list.flatten(), predicted_labels.flatten())

    # Plot confusion matrix
    class_names = ['Normal', 'Anomaly']
    sns.set()
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)

    t = f1_score(gt_list.flatten(), predicted_labels.flatten())

    plt.title('Confusion Matrix (F1 score = %.3f)' % t)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, f'{class_name}_confusion_matrix.png'))
    plt.close()

    # class, image ROCAUC, pixel ROCAUC, inference_time
    # with open(args.save_path + '/' + f'lst.txt', "a") as f:
    #     f.write('{}-{:.3f}-{:.3f}-{:.3f}\n'.format(class_name, img_roc_auc, per_pixel_rocauc, inference_time))

    # plot test images and detection
    plot_fig(img_list, scores, gt_mask_list, pixel_threshold, save_dir, class_name, img_threshold_optimal,
             img_scores, gt_list)

# Function to load the pre-trained model
def load_model(model, checkpoint, gpu, device):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()
        # Special treatment if multiple GPUs were used to train VQ-VAE
        if gpu:
            new_state_dict = OrderedDict()

            for k, v in ckpt.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            # load params
            model.load_state_dict(new_state_dict)

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

        if 'model' in ckpt:
            ckpt = ckpt['model']

        model.load_state_dict(ckpt)

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

    return model

# Main function to execute the anomaly detection pipeline
if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument("--resize", type=int, default=292)
    parser.add_argument("--centerCrop", type=int, default=256)
    parser.add_argument("--randomCrop", type=int, default=282)
    parser.add_argument("--randomRotation", type=int, default=2)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--class_name', type=str)

    args = parser.parse_args()

    model_vqvae = load_model('vqvae', args.vqvae, args.gpu, device)
    model_top = load_model('pixelsnail_top', args.top, args.gpu, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, args.gpu, device)

    test_dataset = mvtec.MVTecDataset(args.data_path, class_name=args.class_name, randomRotation=args.randomRotation,
                                      resize=args.resize, randomCrop=args.randomCrop, centerCrop=args.centerCrop,
                                      is_train=False)

    # Setting num_workers=0 and pin_memory=True seems to have enabled me to debug the code; otherwise, the variables
    # would not load
    loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    extract(loader, model_vqvae, device, model_top, model_bottom, args.batch, [32, 32], [64, 64], temperature=args.temp,
            class_name=args.class_name)