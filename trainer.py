import torch
import torch.nn.functional as F
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF  

# Disable cuDNN
cudnn.enabled = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def one_hot_encode(tensor, num_classes):
    return F.one_hot(tensor, num_classes=num_classes).permute(0, 3, 1, 2).float()

def calculate_metrics(pred, target, num_classes=33, epsilon=1e-6):
    pred_one_hot = one_hot_encode(torch.argmax(pred, dim=1), num_classes)
    target_one_hot = one_hot_encode(target, num_classes)

    intersection = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    union = pred_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    iou_score = (intersection + epsilon) / (union - intersection + epsilon)

    return dice_score.mean().item(), iou_score.mean().item()

class MDiceLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-6):
        super(MDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred_one_hot = one_hot_encode(torch.argmax(pred, dim=1), self.num_classes)
        target_one_hot = one_hot_encode(target, self.num_classes)

        intersection = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3))
        union = pred_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

        dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)
        mdice_loss = 1 - dice_score.mean()

        return mdice_loss

class HybridLoss(nn.Module):
    def __init__(self, base_loss, mdice_loss, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.base_loss = base_loss
        self.mdice_loss = mdice_loss
        self.alpha = alpha

    def forward(self, pred, target):
        base_loss_value = self.base_loss(pred, target)
        mdice_loss_value = self.mdice_loss(pred, target)
        hybrid_loss_value = self.alpha * base_loss_value + (1 - self.alpha) * mdice_loss_value
        return hybrid_loss_value


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device, scheduler=None, num_epochs=500, num_classes=33):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.train_losses = []
        self.val_losses = []
        self.avg_dice_scores = []
        self.avg_iou_scores = []
        self.scaler = GradScaler()

    def resize_input(self, images, target_size=(224, 224)):
        return TF.resize(images, size=target_size)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for images, masks in tqdm(self.train_loader, leave=True, desc=f"Epoch [{epoch+1}/{self.num_epochs}]"):
            images = self.resize_input(images)
            images = images.to(self.device)
            masks = masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)

            torch.cuda.empty_cache()  # Clear CUDA cache during training

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        dice_scores = []
        iou_scores = []
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = self.resize_input(images)
                images = images.to(self.device)
                masks = masks.to(self.device, dtype=torch.long)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item() * images.size(0)

                dice_score, iou_score = calculate_metrics(outputs, masks, self.num_classes)
                dice_scores.append(dice_score)
                iou_scores.append(iou_score)
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        return epoch_loss, avg_dice, avg_iou

    def train(self, num_epochs=500, patience=50):
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        early_stop_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss, avg_dice, avg_iou = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.avg_dice_scores.append(avg_dice)
            self.avg_iou_scores.append(avg_iou)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            torch.cuda.empty_cache()  # Clear CUDA cache at the end of each epoch

        self.model.load_state_dict(best_model_wts)

    def test(self):
        self.model.eval()
        dice_scores = []
        iou_scores = []
        with torch.no_grad():
            for images, masks in self.test_loader:
                images = self.resize_input(images)
                images = images.to(self.device)
                masks = masks.to(self.device, dtype=torch.long)
                outputs = self.model(images)
                dice_score, iou_score = calculate_metrics(outputs, masks, self.num_classes)
                dice_scores.append(dice_score)
                iou_scores.append(iou_score)
                
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        print(f"Test Dice: {avg_dice:.4f}, Test IoU: {avg_iou:.4f}")

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.avg_dice_scores, label='Dice Score')
        plt.plot(epochs, self.avg_iou_scores, label='IoU Score')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.show()
