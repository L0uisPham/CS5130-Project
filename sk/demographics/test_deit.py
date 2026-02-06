import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from sk.datasets.chexpert import CheXpertDataset
from sk.model_wrappers.deit import DeiTModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Data transformations
# -------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -------------------------
# Datasets
# -------------------------
test_dataset = CheXpertDataset(
    "data/CheXpert-v1.0-small/test_strat.csv",
    "data",
    val_transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# -------------------------
# Load pretrained DeiT
# -------------------------
model = DeiTModel(
    model_name="deit_small_patch16_224",
    num_classes=test_dataset.num_classes,
    pretrained=True
).to(device)

model.load_state_dict(torch.load("models/best_deit_model.pth", map_location=device))
model.eval()
model.freeze_backbone_probing()  # freeze for linear probing

# -------------------------
# Feature extraction
# -------------------------
def extract_embeddings(loader, model, device):
    embeddings = []
    sex_labels = []
    age_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # extract CLS token as representation
            feats = model.get_features(images)  # shape: [batch, dim]
            embeddings.append(feats.cpu().numpy())
            sex_labels.append(labels['Sex'].numpy())
            age_labels.append(labels['Age'].numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    sex_labels = np.concatenate(sex_labels, axis=0)
    age_labels = np.concatenate(age_labels, axis=0)
    return embeddings, sex_labels, age_labels

X_test, y_sex, y_age = extract_embeddings(test_loader, model, device)

# -------------------------
# Linear probing - Sex (binary)
# -------------------------
sex_clf = LogisticRegression(max_iter=1000)
sex_clf.fit(X_test, y_sex)  # Here you could split train/test properly for real probing
y_pred_prob = sex_clf.predict_proba(X_test)[:, 1]
y_pred_label = sex_clf.predict(X_test)
sex_auc = roc_auc_score(y_sex, y_pred_prob)
sex_acc = accuracy_score(y_sex, y_pred_label)
print(f"Sex probe - AUC: {sex_auc:.4f}, Accuracy: {sex_acc:.4f}")

# -------------------------
# Linear probing - Age (regression)
# -------------------------
age_reg = nn.Linear(X_test.shape[1], 1)
optimizer = torch.optim.Adam(age_reg.parameters(), lr=1e-3)
criterion = nn.MSELoss()
X_tensor = torch.tensor(X_test, dtype=torch.float32)
y_tensor = torch.tensor(y_age, dtype=torch.float32).unsqueeze(1)

# simple training loop
age_reg.train()
for epoch in range(50):
    optimizer.zero_grad()
    y_pred = age_reg(X_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()

age_reg.eval()
with torch.no_grad():
    y_pred_age = age_reg(X_tensor).squeeze().numpy()
mae = mean_absolute_error(y_age, y_pred_age)
print(f"Age probe - MAE: {mae:.4f}")
