"""
PhishGuard AI - FastAPI Backend
Serves XGBoost + LSTM + CharCNN ensemble predictions for phishing URL detection.
"""

import json
import os
import re
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phishguard")

BASE_DIR = os.path.dirname(__file__)

# ──────────────────────────────────────────────
# Model state (loaded once at startup)
# ──────────────────────────────────────────────
_models: dict = {}

XGB_FEATURE_COLS = [
    "UF2","UF3","UF4","UF5","UF6","UF8","UF10","UF11","UF12","UF13",
    "UF14","UF15","UF16","UF17","UF18","UF19","UF20","UF21","UF22",
    "UF23","UF24","UF25","UF26",
]

# CharCNN architecture (inferred from state_dict tensor shapes)
CNN_MAX_LEN = 200
CNN_EMB_DIM = 64


# ──────────────────────────────────────────────
# CharCNN model definition
# ──────────────────────────────────────────────
def build_char_cnn(vocab_size: int):
    """
    Architecture inferred from state_dict:
      Embedding(vocab_size, 64)
      Conv1d(64→128, k=5) + ReLU + GlobalMaxPool
      Conv1d(128→128, k=3) + ReLU + GlobalMaxPool
      Conv1d(128→64,  k=3) + ReLU + GlobalMaxPool
      Linear(64, 64)  + ReLU
      Linear(64, 1)   + Sigmoid
    """
    import torch
    import torch.nn as nn

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, CNN_EMB_DIM, padding_idx=0)
            self.conv1 = nn.Conv1d(CNN_EMB_DIM, 128, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            # x: (B, L)
            e = self.embedding(x).permute(0, 2, 1)   # (B, 64, L)
            e = self.relu(self.conv1(e)).max(dim=2)[0]  # (B, 128)
            # need 3-dim for conv2
            e = e.unsqueeze(2)
            e = self.relu(self.conv2(e)).max(dim=2)[0]  # (B, 128)
            e = e.unsqueeze(2)
            e = self.relu(self.conv3(e)).max(dim=2)[0]  # (B, 64)
            e = self.relu(self.fc1(e))
            return torch.sigmoid(self.fc2(e)).squeeze(1)

    return CharCNN()


# ──────────────────────────────────────────────
# Startup / shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── XGBoost ──────────────────────────────
    try:
        xgb_model = joblib.load(os.path.join(BASE_DIR, "0623_augmented_xgb_model.pkl"))
        xgb_cfg   = joblib.load(os.path.join(BASE_DIR, "0623_augmented_model_config.pkl"))
        xgb_threshold = (
            float(xgb_cfg.get("best_threshold", xgb_cfg.get("threshold", 0.35)))
            if isinstance(xgb_cfg, dict) else 0.35
        )
        _models["xgb"] = {"model": xgb_model, "threshold": xgb_threshold}
        logger.info(f"XGBoost loaded (threshold={xgb_threshold})")
    except Exception as e:
        logger.warning(f"XGBoost not loaded: {e}")

    # ── LSTM ─────────────────────────────────
    try:
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "0623_lstm_model.keras"))
        with open(os.path.join(BASE_DIR, "0623_lstm_config.json")) as f:
            lstm_cfg = json.load(f)
        lstm_threshold = float(lstm_cfg.get("threshold", 0.45))
        _models["lstm"] = {"model": lstm_model, "threshold": lstm_threshold}
        logger.info(f"LSTM loaded (threshold={lstm_threshold})")
    except Exception as e:
        logger.warning(f"LSTM not loaded: {e}")

    # ── CharCNN (ONNX Runtime) ────────────────
    try:
        import onnxruntime as ort
        with open(os.path.join(BASE_DIR, "tokenizer_CNN.pkl"), "rb") as f:
            char2idx = pickle.load(f)
        sess = ort.InferenceSession(
            os.path.join(BASE_DIR, "cnn_model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        _models["cnn"] = {"session": sess, "char2idx": char2idx, "threshold": 0.4}
        logger.info("CharCNN (ONNX) loaded")
    except Exception as e:
        logger.warning(f"CharCNN not loaded: {e}")

    if not _models:
        raise RuntimeError("No models could be loaded — aborting startup")

    yield
    _models.clear()


app = FastAPI(
    title="PhishGuard AI API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────
from transfer import extract_url_features   # noqa: E402  (local module)


def _clean_for_lstm(url: str) -> str:
    url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    return url.strip().lower().encode("ascii", "ignore").decode("ascii")


def _encode_for_cnn(url: str, char2idx: dict, max_len: int = CNN_MAX_LEN):
    import torch
    clean = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).lower()
    ids = [char2idx.get(c, 0) for c in clean[:max_len]]
    # pad
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class URLItem(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def must_be_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty")
        return v


class BatchRequest(BaseModel):
    urls: list[str]

    @field_validator("urls")
    @classmethod
    def limit_batch(cls, v):
        if len(v) > 20:
            raise ValueError("Maximum 20 URLs per request")
        return v


class PredictionResult(BaseModel):
    url: str
    is_phishing: bool
    confidence: float          # 0–1, higher = more likely phishing
    verdict: str               # "safe" | "suspicious" | "phishing"
    scores: dict               # per-model raw probabilities
    reasons: list[str]         # human-readable flags


# ──────────────────────────────────────────────
# Core prediction logic
# ──────────────────────────────────────────────
def _predict_one(url: str) -> PredictionResult:
    scores = {}
    votes = []
    reasons = []

    # ── XGBoost ──────────────────────────────
    if "xgb" in _models:
        try:
            feats = extract_url_features(url)
            x = np.array([[feats.get(c, 0) for c in XGB_FEATURE_COLS]])
            prob = float(_models["xgb"]["model"].predict_proba(x)[0][1])
            scores["xgboost"] = round(prob, 4)
            votes.append(prob >= _models["xgb"]["threshold"])
            _add_reasons(feats, reasons)
        except Exception as e:
            logger.warning(f"XGB predict error: {e}")

    # ── LSTM ─────────────────────────────────
    if "lstm" in _models:
        try:
            import tensorflow as tf
            clean = _clean_for_lstm(url)
            # character-level integer encoding (same as training: ord, clipped to 128)
            ids = [min(ord(c), 127) for c in clean[:200]]
            ids = ids + [0] * (200 - len(ids))
            x = tf.constant([ids], dtype=tf.int32)
            prob = float(_models["lstm"]["model"].predict(x, verbose=0)[0][0])
            scores["lstm"] = round(prob, 4)
            votes.append(prob >= _models["lstm"]["threshold"])
        except Exception as e:
            logger.warning(f"LSTM predict error: {e}")

    # ── CharCNN (ONNX) ────────────────────────
    if "cnn" in _models:
        try:
            char2idx = _models["cnn"]["char2idx"]
            clean = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).lower()
            ids = [char2idx.get(c, 0) for c in clean[:CNN_MAX_LEN]]
            ids = ids + [0] * (CNN_MAX_LEN - len(ids))
            x = np.array([ids], dtype=np.int64)
            sess = _models["cnn"]["session"]
            out = sess.run(None, {"input": x})
            prob = float(out[0][0])
            scores["cnn"] = round(prob, 4)
            votes.append(prob >= _models["cnn"]["threshold"])
        except Exception as e:
            logger.warning(f"CNN predict error: {e}")

    if not scores:
        raise HTTPException(status_code=503, detail="No models available")

    # Ensemble: average probability
    avg_prob = sum(scores.values()) / len(scores)
    is_phishing = sum(votes) > len(votes) / 2   # majority vote

    if avg_prob >= 0.65:
        verdict = "phishing"
    elif avg_prob >= 0.40:
        verdict = "suspicious"
    else:
        verdict = "safe"

    return PredictionResult(
        url=url,
        is_phishing=is_phishing,
        confidence=round(avg_prob, 4),
        verdict=verdict,
        scores=scores,
        reasons=list(set(reasons)),
    )


def _add_reasons(feats: dict, reasons: list):
    flag_map = {
        "UF3":  "使用 IP 位址作為主機名稱",
        "UF4":  "URL 含有 @ 符號",
        "UF10": "使用短網址服務",
        "UF11": "主機名稱含有連字號 (-)",
        "UF12": "含有敏感關鍵字（login / verify / bank…）",
        "UF13": "冒用知名品牌名稱",
        "UF18": "含有 URL 編碼字元 (%)",
        "UF20": "品牌名出現在子網域（非主網域）",
        "UF24": "使用免費虛擬主機服務",
        "UF26": "使用高風險頂級域名（.sbs / .xyz…）",
    }
    for k, msg in flag_map.items():
        if feats.get(k, 0) == 1:
            reasons.append(msg)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.post("/predict", response_model=PredictionResult)
def predict(item: URLItem):
    return _predict_one(item.url)


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(req: BatchRequest):
    return [_predict_one(u) for u in req.urls]
