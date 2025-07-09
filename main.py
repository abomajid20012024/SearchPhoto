import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# إعدادات عامة
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3
MARGIN = 1.0
EMBED_DIM = 128
DATASET_PATH = "images"


# التحويلات
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Dataset لتكوين Triplets
class TripletProductDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.data = {
            cls: [os.path.join(root_dir, cls, img) for img in os.listdir(os.path.join(root_dir, cls))]
            for cls in self.classes
        }

    def __len__(self):
        return 10000  # عدد العينات العشوائية في كل epoch

    def __getitem__(self, index):
        pos_class = random.choice(self.classes)
        neg_class = random.choice([cls for cls in self.classes if cls != pos_class])

        anchor_path, positive_path = random.sample(self.data[pos_class], 2)
        negative_path = random.choice(self.data[neg_class])

        def load_image(p):
            img = Image.open(p).convert("RGB")
            return self.transform(img)

        return load_image(anchor_path), load_image(positive_path), load_image(negative_path)
# نموذج التمثيل Embedding Model
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBED_DIM):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # Remove final FC layer
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# دالة التدريب
def train(model, dataloader, optimizer, loss_fn, epochs=EPOCHS):
    model.train()
    log_data = []

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for anchor, positive, negative in loop:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = loss_fn(anchor_out, positive_out, negative_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"📘 Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

        log_data.append({
            "Epoch": epoch + 1,
            "Loss": avg_loss,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # حفظ ملف Excel
    df = pd.DataFrame(log_data)
    df.to_excel("training_log.xlsx", index=False)
    print("✅ تم حفظ نتائج التدريب في training_log.xlsx")

# MAIN
if __name__ == "__main__":
    print("📥 تحميل البيانات...")
    dataset = TripletProductDataset(DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("🧠 إنشاء النموذج...")
    model = EmbeddingNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.TripletMarginLoss(margin=MARGIN)

    print("🚀 بدء التدريب...")
    train(model, dataloader, optimizer, loss_fn)

    # حفظ النموذج
    torch.save(model.state_dict(), "embedding_model.pth")
    print("✅ تم حفظ النموذج في embedding_model.pth")