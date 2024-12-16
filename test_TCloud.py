import torch
import torch.nn as nn  
from torch.utils.data import DataLoader
from model.networks import WaveCNN_CR
from datasets.datasets import CloudRemovalDataset
from torchvision import transforms
from utils import validation


# ---  hyper-parameters for testing the neural network --- #
test_data_dir = './data/T-Cloud/test/'
test_batch_size = 1
data_threads = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
test_dataset = CloudRemovalDataset(root_dir = test_data_dir, transform = transforms.Compose([transforms.ToTensor()]), train=False)
test_dataloader = DataLoader(test_dataset, batch_size = test_batch_size, num_workers=data_threads, shuffle=False)

# --- Define the network --- #
model = WaveCNN_CR()

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=[0])

# --- Load the network weight --- #
model.load_state_dict(torch.load('./checkpoints/T-Cloud/cloud_removal_epoch_latest.pth'))

# --- Use the evaluation model in testing --- #
model.eval()
print('--- Testing starts! ---')
print('Length of test set:', len(test_dataloader))
test_psnr, test_ssim, test_ciede2000 = validation(model, test_dataloader, device, save_tag=True, save_path='./results/T-Cloud/')
print('test_psnr: {:.2f}, test_ssim: {:.4f}, test_cided2000: {:.4f}'.format(test_psnr, test_ssim, test_ciede2000))

