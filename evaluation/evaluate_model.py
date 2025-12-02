import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import mean_absolute_error
import torchvision.transforms as T

weights_file = "poodle_age_predictor.pt"
images_dir = "../dataset/vanilla_poodles"
orig_dir = "../dataset/matched_poodles"
real_dir = "../dataset/poodle_images"
gen_dir = images_dir
real_age_file = "../dataset/combined.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(weights_path, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def predict_image(model, img_path, device):
    image = Image.open(img_path).convert("RGB")
    image = transform_resnet(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image).item()
    return pred

def predict_directory(model, directory, device):
    results = []
    for fname in os.listdir(directory):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img_path = os.path.join(directory, fname)
            age = predict_image(model, img_path, device)
            results.append({"filename": fname, "predicted_age": age})
    return pd.DataFrame(results)

def extract_age_from_filename(filename):
    match = re.search(r"_(\d+)m", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract age from filename: {filename}")

def evaluate_predictions(df):
    df["target_age"] = df["filename"].apply(extract_age_from_filename)
    mae = mean_absolute_error(df["target_age"], df["predicted_age"])
    return df, mae

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device)
dino.eval()

dino_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def dino_embed(img_path):
    img = Image.open(img_path).convert("RGB")
    x = dino_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0)

def strip_age_suffix(filename):
    return re.sub(r"_(\d+)m(?=\.[A-Za-z]+$)", "", filename)

def find_original_image(gen_filename):
    base = strip_age_suffix(gen_filename)
    base_no_ext = os.path.splitext(base)[0]
    for fname in os.listdir(orig_dir):
        if os.path.splitext(fname)[0] == base_no_ext:
            return os.path.join(orig_dir, fname)
    return None

def variance_of_laplacian(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap.var()

real_age_map = {}
with open(real_age_file, "r") as f:
    for line in f:
        fname, age_str = line.strip().split()
        real_age_map[fname] = int(age_str)

class InceptionFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        inception.fc = nn.Identity()
        if hasattr(inception, "AuxLogits"):
            inception.AuxLogits = nn.Identity()
        self.model = inception.eval().to(device)
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

feature_extractor = InceptionFeatures()

tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_img_tensor(path):
    return tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

def polynomial_mmd_averages(x, y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    Kxx = (gamma * x @ x.T + coef0) ** degree
    Kyy = (gamma * y @ y.T + coef0) ** degree
    Kxy = (gamma * x @ y.T + coef0) ** degree
    m = x.shape[0]
    n = y.shape[0]
    sum_Kxx = (Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1))
    sum_Kyy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
    sum_Kxy = Kxy.mean()
    return sum_Kxx + sum_Kyy - 2 * sum_Kxy

def compute_kid(real_feats, gen_feats, subsets=50, subset_size=50):
    scores = []
    for _ in range(subsets):
        real_idx = np.random.choice(len(real_feats), subset_size, replace=True)
        gen_idx = np.random.choice(len(gen_feats), subset_size, replace=True)
        x = real_feats[real_idx]
        y = gen_feats[gen_idx]
        val = polynomial_mmd_averages(x, y).item()
        scores.append(val)
    return float(np.mean(scores)), float(np.std(scores))

def extract_age_from_filename_local(fname):
    m = re.search(r"_(\d+)m", fname)
    return int(m.group(1)) if m else None

real_paths = [
    os.path.join(real_dir, fname)
    for fname, age in real_age_map.items()
    if 20 <= age <= 44 and os.path.exists(os.path.join(real_dir, fname))
]

gen_paths = [
    os.path.join(gen_dir, f)
    for f in os.listdir(gen_dir)
    if extract_age_from_filename_local(f) == 36
]

def extract_features(paths):
    feats = []
    for p in paths:
        feats.append(feature_extractor(load_img_tensor(p)).cpu())
    return torch.cat(feats, dim=0)

model = load_model(weights_file, device)
df = predict_directory(model, images_dir, device)
df, mae = evaluate_predictions(df)
pd.DataFrame([{"metric": "MAE", "value": mae}]).to_csv("vanilla_mae_results.csv", index=False)

dino_scores = []
for _, row in df.iterrows():
    gname = row["filename"]
    gpath = os.path.join(images_dir, gname)
    orig = find_original_image(gname)
    if orig is None:
        dino_scores.append(np.nan)
        continue
    orig_feat = dino_embed(orig)
    gen_feat = dino_embed(gpath)
    sim = torch.sum(orig_feat * gen_feat).item()
    dino_scores.append(sim)

df["dino_similarity"] = dino_scores

blur_scores = []
for _, row in df.iterrows():
    p = os.path.join(images_dir, row["filename"])
    blur_scores.append(variance_of_laplacian(p))
df["blurriness_laplacian"] = blur_scores

real_feats = extract_features(real_paths)
gen_feats = extract_features(gen_paths)

kid_mean, kid_std = compute_kid(real_feats, gen_feats)

df_kid = pd.DataFrame([{
    "comparison": "real_20_44m_vs_generated_36m",
    "num_real": len(real_paths),
    "num_generated": len(gen_paths),
    "kid_mean": kid_mean,
    "kid_std": kid_std
}])

df.to_csv("vanilla_predicted_ages.csv", index=False)
df_kid.to_csv("vanilla_kid_results.csv", index=False)
