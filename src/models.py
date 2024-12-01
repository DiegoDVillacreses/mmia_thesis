import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchvision.models import resnet50, vgg16, efficientnet_b7
from transformers import ViTForImageClassification

import torchmetrics
import torch.optim as optim
from config import device, PATH_SUPERVISED_MODELS

from copy import deepcopy
from tqdm.notebook import tqdm

class ViTLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, frozen_prop=0.9, weight_decay=0.01):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224'
        )
        # Change classifier output dimension to 1
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)

        # Freeze initial layers
        total_layers = len(list(self.model.parameters()))
        threshold = int(frozen_prop*total_layers)
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = i >= threshold

        # Use BCEWithLogitsLoss for binary classification
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.frozen_prop = frozen_prop

        # Define metrics
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.test_recall = torchmetrics.Recall(task='binary')

    def forward(self, x):
        return self.model(x).logits  # Shape: [batch_size, 1]

    def training_step(self, batch, batch_idx):
        images, labels = batch  # labels shape: [batch_size]
        outputs = self(images).squeeze(1)  # Shape: [batch_size]
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        acc = self.train_acc(preds, labels.int())
        precision = self.train_precision(preds, labels.int())
        recall = self.train_recall(preds, labels.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze(1)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        acc = self.val_acc(preds, labels.int())
        precision = self.val_precision(preds, labels.int())
        recall = self.val_recall(preds, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze(1)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        acc = self.test_acc(preds, labels.int())
        precision = self.test_precision(preds, labels.int())
        recall = self.test_recall(preds, labels.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    




class CNNLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, frozen_prop=0.9, model_name='resnet'):
        super().__init__()

        # Initialize the model based on the model_name
        if model_name == 'resnet':
            self.model = resnet50(weights='ResNet50_Weights.DEFAULT')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 1)
            layers = list(self.model.children())
        elif model_name == 'vgg':
            self.model = vgg16(weights="VGG16_Weights.DEFAULT")
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, 1)
            layers = list(self.model.features) + list(self.model.classifier)
        elif model_name == 'efficientnet':
            self.model = efficientnet_b7(weights="EfficientNet_B7_Weights.DEFAULT")
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, 1)
            layers = list(self.model.features) + [self.model.classifier]
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Freeze initial layers
        total_layers = len(layers)
        threshold = int(frozen_prop * total_layers)
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = i >= threshold

        # Use BCEWithLogitsLoss for binary classification
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.frozen_prop = frozen_prop

        # Define metrics
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.test_recall = torchmetrics.Recall(task='binary')

    def forward(self, x):
        return self.model(x).squeeze(1)  # Shape: [batch_size]

    def training_step(self, batch, batch_idx):
        images, labels = batch  # labels shape: [batch_size]
        outputs = self(images)  # Shape: [batch_size]
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        acc = self.train_acc(preds, labels.int())
        precision = self.train_precision(preds, labels.int())
        recall = self.train_recall(preds, labels.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        acc = self.val_acc(preds, labels.int())
        precision = self.val_precision(preds, labels.int())
        recall = self.val_recall(preds, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        acc = self.test_acc(preds, labels.int())
        precision = self.test_precision(preds, labels.int())
        recall = self.test_recall(preds, labels.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        return {
            'optimizer': optimizer,
            'monitor': 'val_loss'
        }




class LogisticRegression(pl.LightningModule):

    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs*0.6),
                                                                  int(self.hparams.max_epochs*0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')

@torch.no_grad()
def prepare_data_features(model, data_loader):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    #data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels)

def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100_000, patience=50, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(PATH_SUPERVISED_MODELS, "LogisticRegression"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor="val_acc", patience=patience, mode="max")
        ],
        enable_progress_bar=False,
        check_val_every_n_epoch=10
    )
    trainer.logger._default_hp_metric = None

    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=0)

    pl.seed_everything(0)  # To be reproducable
    model = LogisticRegression(**kwargs)
    trainer.fit(model, train_loader, test_loader)
    model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result

def get_smaller_dataset(original_dataset, num_imgs_per_label):
    new_dataset = data.TensorDataset(
        *[t.unflatten(0, (2, -1))[:,:num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors]
    )
    return new_dataset


class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        #self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        self.convnet = torchvision.models.resnet50(num_classes=4*hidden_dim)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
            # drop out
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                            cos_sim.masked_fill(pos_mask, -9e15)],
                            dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')

class EmbeddingExtractor_VIT(nn.Module):
    def __init__(self, vit_model, layer_idx=-1, use_cls_token=True):
        """
        Extract embeddings from the pretrained Vision Transformer (ViT) model.
        
        Args:
            vit_model: An instance of ViTForImageClassification.
            layer_idx (int): Index of the transformer encoder layer to extract embeddings from.
                             Defaults to -1 (last layer).
            use_cls_token (bool): If True, use the CLS token embedding.
                                  If False, use the mean of all patch embeddings.
        """
        super(EmbeddingExtractor_VIT, self).__init__()
        self.vit = vit_model.model  # Access the underlying ViT model
        self.layer_idx = layer_idx
        self.use_cls_token = use_cls_token

    def forward(self, x):
        """
        Forward pass to extract embeddings.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width).
        
        Returns:
            Tensor of shape (batch_size, embedding_dim) representing the selected embeddings.
        """
        outputs = self.vit(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[self.layer_idx]  # Select the desired layer

        if self.use_cls_token:
            # Return the CLS token embedding
            return hidden_states[:, 0]  # Shape: (batch_size, embedding_dim)
        else:
            # Return the mean of all patch embeddings (excluding CLS token)
            return hidden_states[:, 1:].mean(dim=1)  # Shape: (batch_size, embedding_dim)
