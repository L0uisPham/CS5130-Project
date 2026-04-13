import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
from pathlib import Path

from src.ensemble.ensemble import Ensemble
from src.datasets.loaders import build_dataloaders

class LinearProbeModel(nn.Module):
    def __init__(self, backbone, feature_dim, model_name="model"):
        super(LinearProbeModel, self).__init__()
        self.backbone = backbone
        self.model_name = model_name
        
        # Freeze all parameters in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        
            
        # Linear head for Sex (Binary Classification)
        self.sex_head = nn.Linear(feature_dim, 1)
        
        # Linear head for Age (Regression)
        self.age_head = nn.Linear(feature_dim, 1)

    def forward(self, batch):
        with torch.no_grad():
            features = self.backbone(batch).logits

        """out = self.backbone(batch)
        print(type(out))
        print(out)"""
            
        sex_logits = self.sex_head(features)
        age_pred = self.age_head(features)
        
        return sex_logits, age_pred

def train_probe(model, train_loader, val_loader, device, epochs=10):
    model.to(device)
    
    """for batch in train_loader:
        print(type(batch))
        print(batch)
        break"""

    # Loss functions
    criterion_sex = nn.BCEWithLogitsLoss()
    criterion_age = nn.MSELoss()
    
    # Only optimize the heads
    optimizer = optim.Adam(
        list(model.sex_head.parameters()) + list(model.age_head.parameters()), 
        lr=1e-3
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"{model.model_name} - Epoch {epoch+1}"):
            # Assuming your loader returns: images, (pathology_labels, sex_labels, age_labels)
            images = batch.x.to(device)

            target_sex = batch.meta['sex_num'].to(device).float().unsqueeze(1)
            target_age = batch.meta['age'].to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            
            sex_out, age_out = model(batch)
            
            loss_sex = criterion_sex(sex_out, target_sex)
            loss_age = criterion_age(age_out, target_age)
            
            # You can weight these if one dominates the scale
            total_loss = loss_sex + (loss_age * 0.01) 
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        print(f"Epoch {epoch+1} Loss: {train_loss/len(train_loader):.4f}")
    
    return model


def main():
    # 1. Prep your models
    e = Ensemble()
    convnext = e.conv_model
    swin = e.swin_model

    convnext.model.classifier = nn.Sequential(
        *list(convnext.model.classifier.children())[:-1]
    )
    swin.head = nn.Identity()

    probe_conv = LinearProbeModel(convnext, 768, "ConvNeXt")
    probe_swin = LinearProbeModel(swin, 768, "Swin")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = Path("pipeline/configs/chexpert.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    loaders = build_dataloaders(cfg)

    print("Probing ConvNeXt...")
    train_probe(probe_conv, loaders['train'], loaders['val'], device)

    print("Probing Swin...")
    train_probe(probe_swin, loaders['train'], loaders['val'], device)


if __name__ == "__main__":
    main()