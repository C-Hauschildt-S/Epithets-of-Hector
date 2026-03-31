# philta_ctx.py
# PhilTa embeddings + tiny classifier for contextual sentiment (neg/neu/pos)
# Requires: pip install transformers torch scikit-learn joblib

from typing import Optional, Tuple
import os
import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel
import joblib

MODEL_ID = "bowphs/PhilTa"
CLF_PATH = "philta_sentiment.joblib"   # put/produce this file to enable contextual sentiment

_tok = None
_enc = None
_clf = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _lazy_load_encoder():
    global _tok, _enc
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if _enc is None:
        # Use encoder-only to avoid decoder_input_ids requirement
        _enc = T5EncoderModel.from_pretrained(MODEL_ID).to(_device)
        _enc.eval()

def _lazy_load_clf() -> bool:
    global _clf
    if _clf is not None:
        return True
    if os.path.exists(CLF_PATH):
        _clf = joblib.load(CLF_PATH)
        return True
    return False

@torch.no_grad()
def _embed(lines):
    _lazy_load_encoder()
    batch = _tok(lines, return_tensors="pt", padding=True, truncation=True, max_length=256)
    batch = {k: v.to(_device) for k, v in batch.items()}
    out = _enc(**batch, output_hidden_states=True)
    H = out.last_hidden_state  # [B,T,d]
    mask = batch["attention_mask"].unsqueeze(-1)  # [B,T,1]
    v = (H * mask).sum(1) / mask.sum(1)           # mean-pool → [B,d]
    return v.detach().cpu().numpy()

def predict_sentiment(text: str) -> Optional[Tuple[float, str]]:
    """
    Returns (score, label) where label in {"neg","neu","pos"} if classifier available,
    else returns None.
    """
    if not _lazy_load_clf():
        return None
    X = _embed([text])
    proba = _clf.predict_proba(X)[0]  # [neg, neu, pos]
    idx = int(np.argmax(proba))
    label_map = {0: "neg", 1: "neu", 2: "pos"}
    label = label_map.get(idx, "neu")
    conf = float(np.max(proba))
    score = (-1.0 if label=="neg" else (0.0 if label=="neu" else 1.0)) * conf
    return score, label
