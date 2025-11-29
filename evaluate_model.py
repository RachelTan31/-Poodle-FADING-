import os
import re
import cv2
import clip
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_absolute_error
from torchvision import transforms, models

weights_file = r"C:\Users\maryl\Downloads\-Poodle-FADING-\evaluation\poodle_age_predictor.pt"
images_dir = r"C:\Users\maryl\Downloads\-Poodle-FADING-\poodle_outputs"
orig_dir = "generated_training_poodles"
real_dir = "dataset/poodle_images"
gen_dir = images_dir
real_age_file = "dataset/combined.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def load_model(path, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform_resnet(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(img).item()

def predict_directory(model, directory):
    rows = []
    for fname in os.listdir(directory):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            age = predict_image(model, os.path.join(directory, fname))
            rows.append({"filename": fname, "predicted_age": age})
    return pd.DataFrame(rows)

def extract_age_from_filename(fname):
    m = re.search(r"_(\d+)m", fname)
    if not m:
        raise ValueError(fname)
    return int(m.group(1))

def evaluate_predictions(df):
    df["target_age"] = df["filename"].apply(extract_age_from_filename)
    mae = mean_absolute_error(df["target_age"], df["predicted_age"])
    return df, mae

def strip_age_suffix(fname):
    return re.sub(r"_(\d+)m(?=\.[A-Za-z]+$)", "", fname)

def clip_embed(path):
    x = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        v = clip_model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.squeeze(0)

def find_original_image(output_fname):
    base = strip_age_suffix(output_fname)
    base_no_ext = os.path.splitext(base)[0]
    for fname in os.listdir(orig_dir):
        if os.path.splitext(fname)[0] == base_no_ext:
            return os.path.join(orig_dir, fname)
    return None

def compute_clip_similarity(orig_path, out_path):
    e1 = clip_embed(orig_path)
    e2 = clip_embed(out_path)
    return torch.sum(e1 * e2).item()

def variance_of_laplacian(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap.var()

class InceptionFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        inc = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        inc.fc = nn.Identity()
        if hasattr(inc, "AuxLogits"):
            inc.AuxLogits = nn.Identity()
        self.model = inc.eval().to(device)
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

def load_img_tensor(path):
    return transform_inception(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

def polynomial_mmd_averages(x, y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    Kxx = (gamma * x @ x.T + coef0)**degree
    Kyy = (gamma * y @ y.T + coef0)**degree
    Kxy = (gamma * x @ y.T + coef0)**degree
    m = x.shape[0]
    n = y.shape[0]
    sum_Kxx = (Kxx.sum() - Kxx.diagonal().sum()) / (m*(m-1))
    sum_Kyy = (Kyy.sum() - Kyy.diagonal().sum()) / (n*(n-1))
    sum_Kxy = Kxy.mean()
    return sum_Kxx + sum_Kyy - 2*sum_Kxy

def compute_kid(real_feats, gen_feats, subsets=50, subset_size=50):
    scores = []
    for _ in range(subsets):
        ridx = np.random.choice(len(real_feats), subset_size, replace=True)
        gidx = np.random.choice(len(gen_feats), subset_size, replace=True)
        scores.append(polynomial_mmd_averages(real_feats[ridx], gen_feats[gidx]).item())
    return float(np.mean(scores)), float(np.std(scores))

def extract_age_simple(fname):
    m = re.search(r"_(\d+)m", fname)
    return int(m.group(1)) if m else None

real_age_map = {}
with open(real_age_file, "r") as f:
    for line in f:
        fname, age = line.strip().split()
        real_age_map[fname] = int(age)

model = load_model(weights_file, device)
df = predict_directory(model, images_dir)
df, mae = evaluate_predictions(df)
pd.DataFrame([{"metric": "MAE", "value": mae}]).to_csv("mae_results.csv", index=False)

clip_scores = []
for _, r in df.iterrows():
    out_path = os.path.join(images_dir, r["filename"])
    orig_path = find_original_image(r["filename"])
    if orig_path is None:
        clip_scores.append(np.nan)
    else:
        clip_scores.append(compute_clip_similarity(orig_path, out_path))
df["clip_similarity"] = clip_scores

df["blurriness_laplacian"] = [
    variance_of_laplacian(os.path.join(images_dir, r["filename"]))
    for _, r in df.iterrows()
]

inception = InceptionFeatures()

real_paths = [
    os.path.join(real_dir, fname)
    for fname, age in real_age_map.items()
    if 20 <= age <= 44 and os.path.exists(os.path.join(real_dir, fname))
]

gen_paths = [
    os.path.join(gen_dir, f)
    for f in os.listdir(gen_dir)
    if extract_age_simple(f) == 36
]

def extract_features(paths):
    feats = []
    for p in paths:
        feats.append(inception(load_img_tensor(p)).cpu())
    return torch.cat(feats, dim=0)

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

df.to_csv("predicted_ages.csv", index=False)
df_kid.to_csv("kid_results.csv", index=False)