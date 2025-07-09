# search.py
import torch
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from main import EmbeddingNet
import os

# إعداد النموذج
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNet()
model.load_state_dict(torch.load("embedding_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

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

# تحميل قاعدة البيانات
product_vectors = np.load("product_vectors.npy")
with open("product_paths.pkl", "rb") as f:
    product_paths = pickle.load(f)

# صورة المستخدم
query_image = "test_images/3.jpg"  # ضع هنا المسار لصورة المستخدم
query_vector = get_embedding(query_image)

# حساب التشابه
similarities = cosine_similarity([query_vector], product_vectors)[0]
best_index = similarities.argmax()

# اسم الصورة الأقرب
best_match_path = product_paths[best_index]
best_match_name = os.path.basename(best_match_path)
best_score = similarities[best_index]

# جلب أفضل 3 مؤشرات
top_k = 3
top_indices = similarities.argsort()[::-1][:top_k]

print(f"✅ أقرب {top_k} منتجات:")
for i, idx in enumerate(top_indices, 1):
    path = product_paths[idx]
    name = os.path.basename(path)
    score = similarities[idx]
    print(f"{i}. {name}  🔢 درجة التشابه: {score:.4f}")
