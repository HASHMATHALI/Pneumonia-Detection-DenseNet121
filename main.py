# ============================================================
# PNEUMONIA DETECTION USING DENSENET121
# WITH MC DROPOUT + TEMPERATURE SCALING + GRAD-CAM
# ============================================================

# -------------------- 1. IMPORTS --------------------
import os, cv2, torch, random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# -------------------- 2. DEVICE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------- 3. SEED --------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------- 4. DATA PATH --------------------
DATA_DIR = "chest_xray"  # train/val/test folders inside

# -------------------- 5. TRANSFORMS --------------------
train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------- 6. LOAD DATA --------------------
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR,"val"), test_tfms)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR,"test"), test_tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

class_names = train_ds.classes
print("Classes:", class_names)

# -------------------- 7. CLASS WEIGHTS --------------------
class_counts = np.bincount(train_ds.targets)
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# -------------------- 8. MODEL --------------------
model = models.densenet121(pretrained=True)

for p in model.parameters():
    p.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

model = model.to(device)

# -------------------- 9. LOSS & OPTIMIZER --------------------
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

# -------------------- 10. TRAIN --------------------
EPOCHS = 15
train_acc, val_acc = [], []
train_loss, val_loss = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for x,y in tqdm(train_loader):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)

    train_acc.append(correct/total)
    train_loss.append(running_loss/len(train_loader))

    model.eval()
    correct, total, vloss = 0, 0, 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            vloss += loss.item()
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)

    val_acc.append(correct/total)
    val_loss.append(vloss/len(val_loader))
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc {train_acc[-1]:.4f} | Val Acc {val_acc[-1]:.4f}")

# -------------------- 11. PLOTS --------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_acc,label="Train")
plt.plot(val_acc,label="Val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_loss,label="Train")
plt.plot(val_loss,label="Val")
plt.title("Loss")
plt.legend()
plt.show()

# -------------------- 12. TEST EVALUATION --------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

# -------------------- 13. MC DROPOUT --------------------
def mc_dropout_predict(model, x, T=20):
    model.train()
    preds = []
    for _ in range(T):
        preds.append(F.softmax(model(x), dim=1).unsqueeze(0))
    preds = torch.cat(preds)
    return preds.mean(0), preds.var(0)

# -------------------- 14. TEMPERATURE SCALING --------------------
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.T

scaler = TempScaler().to(device)
optimizer_ts = optim.LBFGS([scaler.T], lr=0.01)

def ts_loss():
    loss = 0
    for x,y in val_loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(scaler(logits), y)
    return loss

optimizer_ts.step(ts_loss)
print("Optimal Temperature:", scaler.T.item())

# -------------------- 15. GRAD-CAM --------------------
def grad_cam(model, img, target):
    fmap, grad = [], []

    def f_hook(m,i,o): fmap.append(o)
    def b_hook(m,gi,go): grad.append(go[0])

    h1 = model.features[-1].register_forward_hook(f_hook)
    h2 = model.features[-1].register_backward_hook(b_hook)

    out = model(img)
    out[:,target].backward()

    w = grad[0].mean(dim=(2,3), keepdim=True)
    cam = (w * fmap[0]).sum(1).squeeze()
    cam = torch.relu(cam)
    cam = cam / cam.max()

    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy()

# -------------------- 16. SAVE --------------------
torch.save(model.state_dict(),"pneumonia_densenet121_exact.pth")
print("Model saved successfully")
