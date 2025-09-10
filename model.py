import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn
import torch


class ResNet18Hotdog(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)



class HotdogClassifier(pl.LightningModule):
    def __init__(self, model, lr=0.001):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()  # Changed from BCEWithLogitsLoss
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log test metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)
        
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = ResNet18Hotdog()
    hotdog_module = HotdogClassifier(model)
    print(hotdog_module)

