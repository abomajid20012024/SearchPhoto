# build_index.py
import os
import numpy as np
import torch
import pickle
from PIL import Image
from torchvision import transforms
from main import EmbeddingNet

# إعداد النموذج
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNet()
model.load_state_dict(torch.load("embedding_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# التحويلات
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy()[0]

# قراءة الصور من مجلد المنتجات
product_folder = "imagesVector"
product_vectors = []
product_paths = []

for filename in os.listdir(product_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(product_folder, filename)
        vec = get_embedding(path)
        product_vectors.append(vec)
        product_paths.append(path)

# حفظ النتائج
np.save("product_vectors.npy", np.array(product_vectors))
with open("product_paths.pkl", "wb") as f:
    pickle.dump(product_paths, f)

print(f"✅ تم حفظ {len(product_paths)} تمثيل لصور المنتجات.")
