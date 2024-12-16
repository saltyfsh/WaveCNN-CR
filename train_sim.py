# --- Imports --- #
from __future__ import print_function
import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.networks import WaveCNN_CR_new
from datasets.datasets import CloudRemovalDataset
import time
from torchvision import transforms
from utils import to_psnr, to_ssim_skimage, to_ciede2000, validation, print_train_log, print_test_log, print_train_log_sim, print_test_log_sim
import os
import lr_scheduler

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument("--restart_weights", help='restart_weights', default=[1, 1], type=list)
parser.add_argument("--eta_mins", help='eta_mins', default=[0.0003, 0], type=list)
parser.add_argument("--periods", help='periods', default=[100, 200], type=list)
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[0], type=list)
opt = parser.parse_args()
print(opt)


# ---  hyper-parameters for training and testing the neural network --- #
train_data_dir = './data/T-Cloud/train/'
val_data_dir = './data/T-Cloud/test/'
train_batch_size = opt.batchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = opt.n_GPUs


device_ids = GPUs_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
print('===> Building model')
netG = WaveCNN_CR_new()

# --- Define the loss --- #
criterionL1 = nn.L1Loss()
criterionL1 = criterionL1.to(device)
criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.to(device)


# --- Multi-GPU --- #
netG = netG.to(device)
netG = nn.DataParallel(netG, device_ids=device_ids)


# --- Build optimizer and scheduler --- #
schedulers = []
optimizers = []
optimizer_G = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)

optimizers.append(optimizer_G)

for optimizer in optimizers:
    schedulers.append(
        lr_scheduler.CosineAnnealingRestartCyclicLR(
            optimizer, periods=opt.periods, restart_weights=opt.restart_weights, eta_mins=opt.eta_mins))


# --- Load training data and validation/test data --- #
train_dataset = CloudRemovalDataset(root_dir=train_data_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=data_threads, shuffle=True)

test_dataset = CloudRemovalDataset(root_dir = val_data_dir, transform = transforms.Compose([transforms.ToTensor()]), train=False)
test_dataloader = DataLoader(test_dataset, batch_size = 1, num_workers=data_threads, shuffle=False)

for epoch in range(1, opt.nEpochs + 1):
    epoch_start_time = time.time()
    print("Training...")

    epoch_loss = 0
    psnr_list = []
    ssim_list = []
    ciede2000_list = []
    iteration_start_time = time.time()
    for iteration, inputs in enumerate(train_dataloader,1):

        cloud, gt = Variable(inputs['cloud_image']), Variable(inputs['ref_image'])
        cloud = cloud.to(device)
        gt = gt.to(device)

        netG.train()
        fake_B = netG(cloud)

        # ----- Update G ----- #
        optimizer_G.zero_grad()
        loss_G = criterionL1(fake_B, gt)
        loss_G.backward()
        optimizer_G.step()

        if iteration % 100 == 0:
            print("===>Epoch[{}]({}/{}): Loss_G: {:.5f} lr: {:.7f} Time: {:.2f}s".format(
                epoch, iteration, len(train_dataloader), loss_G.item(),
                optimizers[0].param_groups[0]['lr'], time.time() - iteration_start_time))

            iteration_start_time = time.time()

        # --- To calculate average PSNR, SSIM, CIEDE2000 --- #
        psnr_list.extend(to_psnr(fake_B, gt))
        ssim_list.extend(to_ssim_skimage(fake_B, gt))
        ciede2000_list.extend(to_ciede2000(fake_B, gt))

    train_psnr = sum(psnr_list) / len(psnr_list)
    train_ssim = sum(ssim_list) / len(ssim_list)
    train_ciede2000 = sum(ciede2000_list) / len(ciede2000_list)

    save_checkpoints = './checkpoints/sim/'
    if os.path.isdir(save_checkpoints)== False:
        os.mkdir(save_checkpoints)

    print_train_log_sim(epoch, train_epoch, train_psnr, train_ssim, train_ciede2000, 'T-Cloud', (time.time()-epoch_start_time)/60)

    if epoch % 5 == 0:
        netG.eval()
        test_psnr, test_ssim, test_ciede2000 = validation(netG, test_dataloader, device)
        print_test_log_sim(epoch, train_epoch, test_psnr, test_ssim, test_ciede2000, 'T-Cloud')

        # --- Save the network  --- #
        torch.save(netG.state_dict(), './checkpoints/sim/cloud_removal_epoch_' + str(epoch) + '.pth')

    old_lr = optimizers[0].param_groups[0]['lr']
    for scheduler in schedulers:
        scheduler.step()

    lr = optimizers[0].param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))

# --- Save the latset network  --- #
torch.save(netG.state_dict(), './checkpoints/sim/cloud_removal_epoch_latest.pth')
print("End of training.")
