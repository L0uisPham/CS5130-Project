import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

from sk.datasets.chexpert import CheXpertDataset
from sk.model_wrappers.deit import DeiTModel

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Data transforms
# -------------------------------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

# -------------------------------------------------
# Dataset / Loader
# -------------------------------------------------
test_dataset = CheXpertDataset(
    "data/CheXpert-v1.0-small/test_strat.csv",
    "data",
    val_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
)

# -------------------------------------------------
# Load pretrained DeiT
# -------------------------------------------------
model = DeiTModel(
    model_name="deit_small_patch16_224",
    num_classes=test_dataset.num_classes,
    pretrained=True,
).to(device)

model.load_state_dict(
    torch.load("sk/tuned_models/best_deit_model.pth", map_location=device)
)

model.eval()
model.freeze_backbone_probing()

# -------------------------------------------------
# Feature extraction
# -------------------------------------------------
def extract_embeddings(loader, model, device):
    embeddings = []
    sex_labels = []
    age_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.cpu()

            feats = model.get_features(images)  # [B, D]
            embeddings.append(feats.cpu().numpy())

            # labels = [sex, age]
            sex_labels.append(labels[:, 0].numpy())
            age_labels.append(labels[:, 1].numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    sex_labels = np.concatenate(sex_labels, axis=0)
    age_labels = np.concatenate(age_labels, axis=0)

    return embeddings, sex_labels, age_labels


X, y_sex, y_age = extract_embeddings(test_loader, model, device)

# -------------------------------------------------
# Train / validation split
# -------------------------------------------------
X_tr, X_va, y_sex_tr, y_sex_va, y_age_tr, y_age_va = train_test_split(
    X,
    y_sex,
    y_age,
    test_size=0.3,
    random_state=42,
    stratify=y_sex,
)

# -------------------------------------------------
# Bootstrap confidence intervals
# -------------------------------------------------
def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, alpha=0.95):
    rng = np.random.default_rng(42)
    scores = []

    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    lo = np.percentile(scores, (1 - alpha) / 2 * 100)
    hi = np.percentile(scores, (1 + alpha) / 2 * 100)

    return np.mean(scores), lo, hi

# -------------------------------------------------
# Sex probe (classification)
# -------------------------------------------------
sex_clf = LogisticRegression(max_iter=2000)
sex_clf.fit(X_tr, y_sex_tr)

y_va_prob = sex_clf.predict_proba(X_va)[:, 1]
y_va_pred = sex_clf.predict(X_va)

sex_auc = roc_auc_score(y_sex_va, y_va_prob)
sex_acc = accuracy_score(y_sex_va, y_va_pred)

mean_auc, lo_auc, hi_auc = bootstrap_ci(
    roc_auc_score, y_sex_va, y_va_prob
)

print(
    f"Sex probe - "
    f"AUC: {sex_auc:.4f} "
    f"[{lo_auc:.4f}, {hi_auc:.4f}], "
    f"Accuracy: {sex_acc:.4f}"
)

# -------------------------------------------------
# Age probe (regression)
# -------------------------------------------------
age_reg = nn.Linear(X_tr.shape[1], 1)
optimizer = torch.optim.Adam(age_reg.parameters(), lr=1e-3)
criterion = nn.MSELoss()

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_age_tr, dtype=torch.float32).unsqueeze(1)
X_va_t = torch.tensor(X_va, dtype=torch.float32)
y_va_t = torch.tensor(y_age_va, dtype=torch.float32).unsqueeze(1)

age_reg.train()
for _ in range(100):
    optimizer.zero_grad()
    loss = criterion(age_reg(X_tr_t), y_tr_t)
    loss.backward()
    optimizer.step()

age_reg.eval()
with torch.no_grad():
    y_va_pred = age_reg(X_va_t).squeeze().numpy()

mae = mean_absolute_error(y_age_va, y_va_pred)

mean_mae, lo_mae, hi_mae = bootstrap_ci(
    mean_absolute_error, y_age_va, y_va_pred
)

print(
    f"Age probe - "
    f"MAE: {mae:.4f} "
    f"[{lo_mae:.4f}, {hi_mae:.4f}]"
)

"""
OUTPUT

Sex probe - AUC: 0.5652 [0.3939, 0.7366], Accuracy: 0.7746
Age probe - MAE: 0.2821 [0.2250, 0.3451]
"""