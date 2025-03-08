import logging
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from tqdm import tqdm

from . import settings


class VGG16Classifier(nn.Module):
    def __init__(self, input_shape=(*settings.IMG_SIZE, 3), num_classes=settings.NUM_CLASSES, learning_rate=1e-4):
        super(VGG16Classifier, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Freeze base model layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(self.base_model.features.children())[:17])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes),
        )

        self.optimizer = torch.optim.Adam(list(self.classifier.parameters()), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_step(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(x)

        y_one_hot = torch.zeros(y.size(0), self.num_classes, device=y.device)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)
        y = y_one_hot

        loss = self.criterion(outputs, y)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader, device):

        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                
                outputs = self(x)
                
                # Convert targets
                y_one_hot = torch.zeros(y.size(0), self.num_classes, device=y.device)
                y_one_hot.scatter_(1, y.unsqueeze(1), 1)
                
                loss = self.criterion(outputs, y_one_hot)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return running_loss / len(dataloader), 100 * correct / total
    
    def summary(self):
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")



class CNNClassifier(nn.Module):
    def __init__(self, input_shape=(*settings.IMG_SIZE, 3), num_classes=settings.NUM_CLASSES, learning_rate=1e-4):
        super(CNNClassifier, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),  # 6x6 from 3 max pooling layers (48 -> 24 -> 12 -> 6)
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, self.num_classes),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_step(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(x)
        
        y_one_hot = torch.zeros(y.size(0), self.num_classes, device=y.device)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)
        y = y_one_hot
        
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, dataloader, device):
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                
                # Convert targets
                y_one_hot = torch.zeros(y.size(0), self.num_classes, device=y.device)
                y_one_hot.scatter_(1, y.unsqueeze(1), 1)
                target = y_one_hot
                
                loss = self.criterion(outputs, target)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        
        return running_loss / len(dataloader), 100 * correct / total

    def summary(self):
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")


class __TrainingResults:
    """
    Stores training history, including loss, acc, lr, and epoch info
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
        
    def add_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None, lr=None):
        """
        Adds the results of a single epoch to the stored history
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        if lr is not None:
            self.learning_rates.append(lr)


def train_model(model, train_loader, val_loader=None, num_epochs=10, device=settings.DEVICE):
    """
    Trains the model and records training history
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for the training dataset
        val_loader (optional): DataLoader for the validation dataset. Defaults to None
        num_epochs (int, optional): Number of training epochs. Defaults to 10
    
    Returns:
        tuple: Trained model and TrainingResults instance containing training history
    """
    
    model = model.to(device)
    
    results = __TrainingResults()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # wrap train_loader with tqdm for a progress bar
        prog_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        
        for i, (x, y) in enumerate(prog_bar):
            x, y = x.to(device), y.to(device)
            loss = model.train_step(x, y)
            running_loss += loss
            
            with torch.no_grad():
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            prog_bar.set_postfix(loss=loss)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct / total) * 100        
        cur_lr = model.optimizer.param_groups[0]['lr']
        
        # Validation phase
        val_loss, val_acc = None, None
        if val_loader:
            val_loss, val_acc = model.evaluate(val_loader, device)
            if val_loss is not None:
                model.scheduler.step(val_loss)
        logging.info(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}%"
            +
            (f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}%" if val_loader else "")
            +
            (f" | Lr: {cur_lr}")
        )
        
        results.add_epoch(
            epoch=epoch+1, 
            train_loss=epoch_loss, 
            train_acc=epoch_acc, 
            val_loss=val_loss, 
            val_acc=val_acc,
            lr=cur_lr
        )
    
    return model, results
