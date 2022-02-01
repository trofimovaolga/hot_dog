import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision.models as models

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchsampler import ImbalancedDatasetSampler

import config

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_sampler(train_dataset):
    # count number of positive labels
    pos_labels = 0
    for _, target in train_dataset:
        pos_labels += target
    
    class_count = [len(train_dataset) - pos_labels, pos_labels]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    target_list = torch.tensor(train_dataset.targets)
    class_weights_all = class_weights[target_list]

    sampler = WeightedRandomSampler(class_weights_all,
                                    num_samples=len(class_weights_all),
                                    replacement=True)
    return sampler


def train_model(model, loss, optimizer, scheduler, train_dl, val_dl, num_epochs, exp_num):
    if config.log: 
        run = wandb.init(project='hot_dog', name=str(exp_num) + '_exp_200')
    
    stats = {'train_loss_history': [],
             'val_loss_history': [],
             'hot_dogs': 0,
             'not_hot_dogs': 0}
    
    for epoch in tqdm(range(num_epochs)):
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dl
                model.train()
            else:
                dataloader = val_dl
                model.eval()

            epoch_loss, avg_epoch_loss = 0.0, 0.0

            for inputs, labels in dataloader:
                # count number of images of each label
                stats['hot_dogs'] += torch.sum(labels==0)
                stats['not_hot_dogs'] += torch.sum(labels==1)
                
                inputs = inputs.to(config.device)
                labels = labels.to(config.device, torch.float32).unsqueeze(-1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)

                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                epoch_loss += loss_value.item()
                                
            avg_epoch_loss = epoch_loss / len(dataloader)

            if phase == 'train':
                stats['train_loss_history'].append(avg_epoch_loss)
            else:
                stats['val_loss_history'].append(avg_epoch_loss)
                if config.log: 
                    wandb.log({"lr": optimizer.param_groups[0]['lr']})
                scheduler.step(avg_epoch_loss)
            
            if config.log:
                wandb.log({f"{phase}_loss": avg_epoch_loss})
    return model, stats

@torch.no_grad()
def test_model(model, test_dataloader):
    test_predictions, test_labels = [], []
    model.eval()
    model = model.to(config.device)

    for inputs, labels in test_dataloader:
        inputs = inputs.to(config.device)
        preds = model(inputs)
        test_predictions.append(torch.sigmoid(preds).data.cpu().numpy())
        test_labels.append(labels)

    test_predictions = np.concatenate(test_predictions)
    test_labels = np.concatenate(test_labels)
    return test_predictions, test_labels

# save examples of wrong predictions
def save_imgs(imgs, labels, target=None, exp_num=0):
    ncols = 10
    nrows = np.ceil(len(imgs) / ncols).astype(int)
    plt.figure(figsize=(ncols * 4, nrows * 4))
    
    for i in range(len(imgs)):
        plt.subplot(nrows, ncols, i + 1)
        pic = np.squeeze(np.transpose(imgs[i].numpy(), (1, 2, 0)))
        plt.imshow(np.clip((config.std * pic + config.mean), 0, 1))
        plt.title('{:.3f} / {}'.format(labels[i].numpy(), target[i]))
        plt.subplots_adjust(top=0.8)
    
    filename = str(exp_num) + '_wrong_predictions.jpeg'
    plt.savefig(filename)
    plt.close('all')

# save roc-curve and calculate optimal threshold
def roc_curve_and_threshold(test_labels, test_predictions):
    score = roc_auc_score(test_labels, test_predictions)
    plt.figure()
    f, ax = plt.subplots(1, 1)
    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)
    ix = np.argmax(tpr - fpr)
    best_thresh = thresholds[ix]
    plt.plot(fpr, tpr, label='ROC (area = {:.4})'.format(score))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.legend()
    plt.grid(True)
    plt.close('all')
    return f, best_thresh, score


def get_wrong_preds(test_predictions, test_labels, test_transforms, threshold=0.5):
    file_names = []
    for (_, _, filenames) in os.walk(config.test_dir):
        file_names.extend(filenames)

    imgs, lbls, trgs = [], [], []
    for i in range(len(test_predictions)):
        if (test_predictions[i][0] > threshold) != test_labels[i]:
            if test_labels[i] > threshold:
                dir = 'not_hot_dog'
            else:
                dir = 'hot_dog'
            source_dir = os.path.join(config.test_dir, dir)
            image = Image.open('{}/{}'.format(source_dir, file_names[i]))
            imgs.append(test_transforms(image))
            lbls.append(torch.tensor(test_predictions[i][0]))
            trgs.append(torch.tensor(test_labels[i]))
    return imgs, lbls, trgs


def save_plot(train_arr, val_arr, exp_num, type):
    plt.figure()
    plt.plot(train_arr, label='train_{}'.format(type))
    plt.plot(val_arr, label='val_{}'.format(type))
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.savefig('{}_{}.jpeg'.format(exp_num, type))
    plt.close('all')


def save_cm(test_labels, test_predictions, best_thresh, exp_num):
    plt.figure()
    cm = confusion_matrix(test_labels, test_predictions > best_thresh)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=config.class_names)
    disp.plot(cmap='Blues', colorbar=False)
    disp.figure_.savefig(str(exp_num) + '_cm.jpeg')
    return disp.figure_


def save_bar(classes, values, exp_num):
    fig = plt.figure()
    plt.bar(classes, values)
    plt.savefig(str(exp_num) + '_bar.jpeg')
    plt.close('all')
    return fig

# training for exact loss and train dataloader
def run_experiment(exp_num, train_dl, val_dl, test_dl, loss_func):
    print('Running experiment number', exp_num)

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 1)
    )
    # model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model = model.to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     'min',
                                                     min_lr=1.0e-6,
                                                     threshold=1.0e-3,
                                                     patience=5)

    model, stats = train_model(model,
                               loss_func,
                               optimizer,
                               scheduler,
                               train_dl,
                               val_dl,
                               config.num_epochs,
                               exp_num)
    
    print('...preparing plots...')
    
    save_plot(stats['train_loss_history'], stats['val_loss_history'], exp_num, 'loss')
    b = save_bar(config.class_names, [stats['hot_dogs'], stats['not_hot_dogs']], exp_num)

    test_predictions, test_labels = test_model(model, test_dl)
    roc, best_thresh, score = roc_curve_and_threshold(test_labels, test_predictions)
    roc.savefig(str(exp_num) + '_roc_curve.jpeg')
    cmp = save_cm(test_labels, test_predictions, best_thresh, exp_num)

    imgs, lbls, trgs = get_wrong_preds(test_predictions,
                                       test_labels,
                                       config.test_transforms,
                                       best_thresh)
    
    save_imgs(imgs, lbls, trgs, exp_num)
    
    print('Train loss: {:.4f} Valid loss: {:.4f}'.format(stats['train_loss_history'][-1], stats['val_loss_history'][-1]))
    print('ROC-AUC: {:.4f}'.format(score))

    if config.log:
        wandb.log({"Class distribution": wandb.Image(b)})
        wandb.log({"ROC-AUC": wandb.Image(roc)})
        wandb.log({"Confusion Matrix": wandb.Image(cmp)})
        wandb.finish()


def main():
    if config.log: wandb.login()
    
    set_seed(config.seed)

    train_dataset = ImageFolder(config.train_dir, config.train_transforms)
    val_dataset = ImageFolder(config.val_dir, config.test_transforms)
    test_dataset = ImageFolder(config.test_dir, config.test_transforms)  

    # prepare train dataloaders for different experiments
    train_dataloader_imbalanced = DataLoader(train_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.num_workers)
                                
    imb_sampler = ImbalancedDatasetSampler(train_dataset)
    imb_samp_train_dataloader = DataLoader(train_dataset,
                                           sampler=imb_sampler,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           num_workers=config.num_workers)

    wght_sampler = make_sampler(train_dataset)
    wght_samp_train_dataloader = DataLoader(train_dataset,
                                            sampler=wght_sampler,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=config.num_workers)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers)

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=config.batch_size, 
                                 shuffle=False, 
                                 num_workers=config.num_workers)

    loss = torch.nn.BCEWithLogitsLoss()
    
    # loss with weights
    num_hot_dog = len(os.listdir(os.path.join(config.train_dir, config.class_names[0])))
    num_not_hot_dog = len(os.listdir(os.path.join(config.train_dir, config.class_names[1])))
    pos_weight = torch.tensor(num_hot_dog / num_not_hot_dog)    
    loss_w = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    experiments = [(train_dataloader_imbalanced, loss), 
                   (train_dataloader_imbalanced, loss_w),
                   (imb_samp_train_dataloader, loss),
                   (wght_samp_train_dataloader, loss)]
    
    for i, params in enumerate(experiments):
        train_dataloader, loss_func = params
        run_experiment(i + 1, train_dataloader, val_dataloader, test_dataloader, loss_func)

    print('Finished')


if __name__ == "__main__":
    main()