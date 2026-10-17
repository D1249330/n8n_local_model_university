"""
PhishGuard AI - FastAPI Backend
Serves XGBoost + LSTM + CharCNN ensemble predictions for phishing URL detection.

模型版本 0802：
  XGBoost  0802_augmented_xgb_model.ubj（XGBoost 原生格式）+ 0802_augmented_model_config.pkl（門檻）
  LSTM     0802_lstm_model.keras（模型內含 TextVectorization，輸入是網址字串）+ 0802_lstm_config.json
  CharCNN  cnn_model.onnx（由 char_cnn_model.pth 經 convert_cnn_to_onnx.py 轉出）+ tokenizer_CNN.pkl
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
MODEL_VERSION = "0802"

# ──────────────────────────────────────────────
# Model state (loaded once at startup)
# ──────────────────────────────────────────────
_models: dict = {}

XGB_FEATURE_COLS = [
    "UF2","UF3","UF4","UF5","UF6","UF8","UF10","UF11","UF12","UF13",
    "UF14","UF15","UF16","UF17","UF18","UF19","UF20","UF21","UF22",
    "UF23","UF24","UF25","UF26",
]

# CharCNN 前處理參數（模型結構見 convert_cnn_to_onnx.py；Render 上只用 ONNX，不需要 PyTorch）
CNN_MAX_LEN = 200
CNN_THRESHOLD = 0.4


# ──────────────────────────────────────────────
# Startup / shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── XGBoost ──────────────────────────────
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join(BASE_DIR, "0802_augmented_xgb_model.ubj"))
        xgb_cfg   = joblib.load(os.path.join(BASE_DIR, "0802_augmented_model_config.pkl"))
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
        lstm_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "0802_lstm_model.keras"))
        with open(os.path.join(BASE_DIR, "0802_lstm_config.json")) as f:
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
        _models["cnn"] = {"session": sess, "char2idx": char2idx, "threshold": CNN_THRESHOLD}
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


_SCHEME_RE = re.compile(r"^\s*https?://", re.IGNORECASE)


def _strip_scheme(url: str) -> str:
    """訓練資料的網址都沒有 http:// / https://，預測前先拿掉，三個模型都用這個版本。"""
    return _SCHEME_RE.sub("", url.strip()).strip()


def _clean_for_lstm(bare: str) -> str:
    return bare.lower().encode("ascii", "ignore").decode("ascii")


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
    bare = _strip_scheme(url)          # 送進模型的網址（無協定）；回傳時仍用原始 url
    if not bare:
        raise HTTPException(status_code=422, detail="url must not be empty")
    scores = {}
    votes = []
    reasons = []

    # ── XGBoost ──────────────────────────────
    if "xgb" in _models:
        try:
            import pandas as pd
            feats = extract_url_features(bare)
            # 用 DataFrame 帶欄位名稱，模型會檢查特徵名稱與順序
            x = pd.DataFrame([[feats.get(c, 0) for c in XGB_FEATURE_COLS]], columns=XGB_FEATURE_COLS)
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
            clean = _clean_for_lstm(bare)
            # 模型第一層是 TextVectorization（逐字元切分、長度 200），直接餵網址字串
            x = tf.constant([[clean]], dtype=tf.string)
            prob = float(_models["lstm"]["model"].predict(x, verbose=0)[0][0])
            scores["lstm"] = round(prob, 4)
            votes.append(prob >= _models["lstm"]["threshold"])
        except Exception as e:
            logger.warning(f"LSTM predict error: {e}")

    # ── CharCNN (ONNX) ────────────────────────
    if "cnn" in _models:
        try:
            char2idx = _models["cnn"]["char2idx"]
            clean = bare.lower()
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
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "models_loaded": list(_models.keys()),
        "thresholds": {k: v.get("threshold") for k, v in _models.items()},
    }


@app.post("/predict", response_model=PredictionResult)
def predict(item: URLItem):
    return _predict_one(item.url)


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(req: BatchRequest):
    return [_predict_one(u) for u in req.urls]
