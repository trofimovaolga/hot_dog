import os
import torch
import torchvision.transforms as transforms

# for wandb logging
log = False

root_dir = os.path.join(os.getcwd(), 'seefood/')
train_dir = os.path.join(root_dir, 'train/')
test_dir = os.path.join(root_dir, 'test/')
val_dir = os.path.join(root_dir, 'val/')

class_names = ['hot_dog', 'not_hot_dog']

seed = 24
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
batch_size = 64
num_epochs = 20
num_workers = 4
lr = 1.0e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])