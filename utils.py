# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import metrics
from torch.autograd import Variable
import os
import numpy as np


def to_psnr(cloud, gt):
    mse = F.mse_loss(cloud, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(cloud, gt):
    cloud_list = torch.split(cloud, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    cloud_list_np = [cloud_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(cloud_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(cloud_list))]
    #ssim_list = [metrics.structural_similarity(cloud_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(cloud_list))]
    ssim_list = [metrics.structural_similarity(cloud_list_np[ind],  gt_list_np[ind], data_range=1, channel_axis=2) for ind in range(len(cloud_list))]

    return ssim_list


def to_ciede2000(cloud, gt):
    cloud_list = torch.split(cloud, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    cloud_list_np = [cloud_list[ind].data.cpu().numpy().squeeze() for ind in range(len(cloud_list))]
    gt_list_np = [gt_list[ind].data.cpu().numpy().squeeze() for ind in range(len(cloud_list))]
    ciede2000_list = [ciede2000(cloud_list_np[ind],  gt_list_np[ind]) for ind in range(len(cloud_list))]

    return ciede2000_list


def validation(net, val_data_loader, device, save_tag=False, save_path='./results/'):

    psnr_list = []
    ssim_list = []
    ciede2000_list = []

    for batch_id, val_data in enumerate(val_data_loader):


        with torch.no_grad():
            cloud, gt, image_name = Variable(val_data['cloud_image']), Variable(val_data['ref_image']),val_data['image_name']
            cloud = cloud.to(device)
            gt = gt.to(device)
            cloud_removal = net(cloud)
            cloud_removal = torch.clip(cloud_removal, 0, 1)

            # --- Calculate the average PSNR --- #
            psnr = to_psnr(cloud_removal, gt)

            # --- Calculate the average SSIM --- #
            ssim = to_ssim_skimage(cloud_removal, gt)

            # --- Calculate the average CIEDE2000 --- #
            ciede2000 = to_ciede2000(cloud_removal, gt)


        psnr_list.extend(psnr)
        ssim_list.extend(ssim)
        ciede2000_list.extend(ciede2000)

        # --- Save image --- #
        if save_tag:
            path = save_path
            if not os.path.exists(path):
                os.makedirs(path)
            save_image(cloud_removal, image_name, path)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    avr_ciede2000 = sum(ciede2000_list) / len(ciede2000_list)

    return avr_psnr, avr_ssim, avr_ciede2000


def save_image(cloud_removal, image_name, path):
    cloud_removal = torch.split(cloud_removal, 1, dim=0)
    batch_num = len(cloud_removal)
    for ind in range(batch_num):
        utils.save_image(cloud_removal[ind], path+'{}'.format(image_name[ind][:-3] + 'png'))


def print_train_log(epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, category, use_time):
    print('Epoch [{}/{}], Train_PSNR:{:.2f}, Train_SSIM:{:.4f}, Train_CIEDE2000:{:.4f}, Time:{:.2f}min'
          .format(epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, use_time))

    # --- Write the training log --- #
    with open('./logs/{}_log.txt'.format(category), 'a') as f:
        print('Date: {}s, Epoch: [{}/{}], Train_PSNR: {:.2f}, Train_SSIM: {:.4f}, Train_CIEDE2000:{:.4f}, Time:{:.2f}min'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, use_time), file=f)
        
def print_train_log_sim(epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, category, use_time):
    print('Epoch [{}/{}], Train_PSNR:{:.2f}, Train_SSIM:{:.4f}, Train_CIEDE2000:{:.4f}, Time:{:.2f}min'
          .format(epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, use_time))

    # --- Write the training log --- #
    with open('./logs/sim/{}_log.txt'.format(category), 'a') as f:
        print('Date: {}s, Epoch: [{}/{}], Train_PSNR: {:.2f}, Train_SSIM: {:.4f}, Train_CIEDE2000:{:.4f}, Time:{:.2f}min'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, use_time), file=f)
        
def print_train_log_color(epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, category, use_time):
    print('Epoch [{}/{}], Train_PSNR:{:.2f}, Train_SSIM:{:.4f}, Train_CIEDE2000:{:.4f}, Time:{:.2f}min'
          .format(epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, use_time))

    # --- Write the training log --- #
    with open('./logs/color/{}_log.txt'.format(category), 'a') as f:
        print('Date: {}s, Epoch: [{}/{}], Train_PSNR: {:.2f}, Train_SSIM: {:.4f}, Train_CIEDE2000:{:.4f}, Time:{:.2f}min'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, use_time), file=f)
        
def print_train_log_sc(epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, category, use_time):
    print('Epoch [{}/{}], Train_PSNR:{:.2f}, Train_SSIM:{:.4f}, Train_CIEDE2000:{:.4f}, Time:{:.2f}min'
          .format(epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, use_time))

    # --- Write the training log --- #
    with open('./logs/sc/{}_log.txt'.format(category), 'a') as f:
        print('Date: {}s, Epoch: [{}/{}], Train_PSNR: {:.2f}, Train_SSIM: {:.4f}, Train_CIEDE2000:{:.4f}, Time:{:.2f}min'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, train_psnr, train_ssim, train_ciede2000, use_time), file=f)


def print_test_log(epoch, num_epochs, test_psnr, test_ssim, test_ciede2000, category):
    print('Epoch [{}/{}], Test_PSNR:{:.2f}, Test_SSIM:{:.4f}, Test_CIEDE2000:{:.4f}'
          .format(epoch, num_epochs, test_psnr, test_ssim, test_ciede2000))

    # --- Write the training log --- #
    with open('./logs/{}_log.txt'.format(category), 'a') as f:
        print('Date: {}s, Epoch: [{}/{}], Test_PSNR: {:.2f}, Test_SSIM: {:.4f}, Test_CIEDE2000:{:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, test_psnr, test_ssim, test_ciede2000), file=f)
        
def print_test_log_sim(epoch, num_epochs, test_psnr, test_ssim, test_ciede2000, category):
    print('Epoch [{}/{}], Test_PSNR:{:.2f}, Test_SSIM:{:.4f}, Test_CIEDE2000:{:.4f}'
          .format(epoch, num_epochs, test_psnr, test_ssim, test_ciede2000))

    # --- Write the training log --- #
    with open('./logs/sim/{}_log.txt'.format(category), 'a') as f:
        print('Date: {}s, Epoch: [{}/{}], Test_PSNR: {:.2f}, Test_SSIM: {:.4f}, Test_CIEDE2000:{:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, test_psnr, test_ssim, test_ciede2000), file=f)
        
def print_test_log_color(epoch, num_epochs, test_psnr, test_ssim, test_ciede2000, category):
    print('Epoch [{}/{}], Test_PSNR:{:.2f}, Test_SSIM:{:.4f}, Test_CIEDE2000:{:.4f}'
          .format(epoch, num_epochs, test_psnr, test_ssim, test_ciede2000))

    # --- Write the training log --- #
    with open('./logs/color/{}_log.txt'.format(category), 'a') as f:
        print('Date: {}s, Epoch: [{}/{}], Test_PSNR: {:.2f}, Test_SSIM: {:.4f}, Test_CIEDE2000:{:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, test_psnr, test_ssim, test_ciede2000), file=f)
        
def print_test_log_sc(epoch, num_epochs, test_psnr, test_ssim, test_ciede2000, category):
    print('Epoch [{}/{}], Test_PSNR:{:.2f}, Test_SSIM:{:.4f}, Test_CIEDE2000:{:.4f}'
          .format(epoch, num_epochs, test_psnr, test_ssim, test_ciede2000))

    # --- Write the training log --- #
    with open('./logs/sc/{}_log.txt'.format(category), 'a') as f:
        print('Date: {}s, Epoch: [{}/{}], Test_PSNR: {:.2f}, Test_SSIM: {:.4f}, Test_CIEDE2000:{:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, test_psnr, test_ssim, test_ciede2000), file=f)



# Converts RGB pixel array to XYZ format.
# Implementation derived from http://www.easyrgb.com/en/math.php
def rgb2xyz(rgb):
    def format(c):
        c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
        return c * 100
    rgb = list(map(format, rgb))
    xyz = [None, None, None]
    xyz[0] = rgb[0] * 0.4124 + rgb[1] * 0.3576 + rgb[2] * 0.1805
    xyz[1] = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
    xyz[2] = rgb[0] * 0.0193 + rgb[1] * 0.1192 + rgb[2] * 0.9505
    return xyz

# Converts XYZ pixel array to LAB format.
# Implementation derived from http://www.easyrgb.com/en/math.php
def xyz2lab(xyz):
    def format(c):
        c = np.where(c > 0.008856, c ** (1. / 3.), (7.787 * c) + (16. / 116.))
        return c
    xyz[0] = xyz[0] / 95.047
    xyz[1] = xyz[1] / 100.00
    xyz[2] = xyz[2] / 108.883
    xyz = list(map(format, xyz))
    lab = [None, None, None]
    lab[0] = (116. * xyz[1]) - 16.
    lab[1] = 500. * (xyz[0] - xyz[1])
    lab[2] = 200. * (xyz[1] - xyz[2])
    return lab

# Converts RGB pixel array into LAB format.
def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

# Returns CIEDE2000 comparison results of two LAB formatted colors.
# Implementation derived from the excel spreadsheet provided here: http://www2.ece.rochester.edu/~gsharma/ciede2000/
def ciede2000(img1, img2):

    img1 = img1.reshape(3,-1)
    img2 = img2.reshape(3,-1)

    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)

    L1 = lab1[0]
    A1 = lab1[1]
    B1 = lab1[2]
    L2 = lab2[0]
    A2 = lab2[1]
    B2 = lab2[2]
    C1 = np.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = np.sqrt((A2 ** 2.) + (B2 ** 2.))
    aC1C2 = np.average([C1, C2], axis=0)
    G = 0.5 * (1. - np.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25. ** 7.))))
    a1P = (1. + G) * A1
    a2P = (1. + G) * A2
    c1P = np.sqrt((a1P ** 2.) + (B1 ** 2.))
    c2P = np.sqrt((a2P ** 2.) + (B2 ** 2.))

    h1P = np.where((a1P==0)&(B1==0), np.zeros_like(a1P), np.degrees(np.arctan2(B1, a1P)) + 360.)
    h1P = np.where((~((a1P==0)&(B1==0)))&(B1>=0), np.degrees(np.arctan2(B1, a1P)), h1P)
    h2P = np.where((a2P==0)&(B2==0), np.zeros_like(a2P), np.degrees(np.arctan2(B2, a2P)) + 360.)
    h2P = np.where((~((a2P==0)&(B2==0)))&(B2>=0), np.degrees(np.arctan2(B2, a2P)), h2P)

    dLP = L2 - L1
    dCP = c2P - c1P

    dhC = np.where(h2P - h1P > 180, np.ones_like(h2P), np.zeros_like(h2P))
    dhC = np.where(h2P - h1P < -180, 2 * np.ones_like(h2P), dhC)
    dhP = np.where(dhC == 0, h2P - h1P, h2P + 360. - h1P)
    dhP = np.where(dhC == 1, h2P - h1P - 360., dhP)

    dHP = 2. * np.sqrt(c1P * c2P) * np.sin(np.radians(dhP / 2.))
    aL = np.average([L1, L2], axis=0)
    aCP = np.average([c1P, c2P], axis=0)

    haC = np.where(c1P*c2P==0, 3*np.ones_like(c1P), 2*np.ones_like(c1P))
    haC = np.where((c1P*c2P!=0)&(np.absolute(h2P-h1P)<=180), np.zeros_like(c1P), haC)
    haC = np.where((c1P*c2P!=0)&(np.absolute(h2P-h1P)>180)&(h2P+h1P<360), np.ones_like(c1P), haC)

    haP = np.average([h1P, h2P], axis=0)
    aHP = np.where(haC==3, h1P + h2P, haP - 180)
    aHP = np.where(haC==0, haP, aHP)
    aHP = np.where(haC==1, haP + 180, aHP)

    lPa50 = (aL - 50) ** 2.
    sL = 1. + (0.015 * lPa50 / np.sqrt(20. + lPa50))
    sC = 1. + 0.045 * aCP
    T = 1. - 0.17 * np.cos(np.radians(aHP - 30.)) + 0.24 * np.cos(np.radians(2. * aHP)) + 0.32 * np.cos(np.radians(3. * aHP + 6.)) - 0.2 * np.cos(np.radians(4. * aHP - 63.))
    sH = 1. + 0.015 * aCP * T
    dTheta = 30. * np.exp(-1. * ((aHP - 275.) / 25.) ** 2.)
    rC = 2. * np.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))
    rT = -np.sin(np.radians(2. * dTheta)) * rC
    fL = dLP / sL / 1.
    fC = dCP / sC / 1.
    fH = dHP / sH / 1.
    dE2000 = np.sqrt(fL ** 2. + fC ** 2. + fH ** 2. + rT * fC * fH)
    return dE2000.mean()

