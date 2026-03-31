# train_philta_sentiment.py
# pip install transformers torch scikit-learn joblib pandas
import pandas as pd, numpy as np, joblib, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, T5EncoderModel

MODEL_ID = "bowphs/PhilTa"
CSV_PATH = "sentiment_train.csv"       # your labeled data (text,label)
OUT_PATH = "philta_sentiment.joblib"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
enc = T5EncoderModel.from_pretrained(MODEL_ID)  # encoder-only
enc.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = enc.to(device)

@torch.no_grad()
def embed(lines):
    batch = tok(list(lines), return_tensors="pt", padding=True, truncation=True, max_length=256)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = enc(**batch, output_hidden_states=True)
    H = out.last_hidden_state                        # [B, T, d_model]
    mask = batch["attention_mask"].unsqueeze(-1)     # [B, T, 1]
    v = (H * mask).sum(1) / mask.sum(1)              # mean-pool → [B, d_model]
    return v.cpu().numpy()

df = pd.read_csv(CSV_PATH)
label_map = {"neg":0, "neu":1, "pos":2}
y = df["label"].map(label_map).values
X = embed(df["text"].tolist())

from collections import Counter

counts = Counter(y)
min_class = min(counts.values())
print("Label counts:", counts)

if min_class < 2 or len(y) < 10:
    print("[WARN] Too few samples per class for a stratified split."
          " Training on ALL data and skipping validation.")
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(X, y)
    joblib.dump(clf, OUT_PATH)
    print("Saved", OUT_PATH)
    raise SystemExit(0)

# else: safe to stratify
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=2000, multi_class="auto")
clf.fit(Xtr, ytr)

yp = clf.predict(Xte)
print(classification_report(yte, yp, target_names=["neg","neu","pos"]))

joblib.dump(clf, OUT_PATH)
print("Saved", OUT_PATH)
