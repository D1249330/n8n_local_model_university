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
LSTM_THRESHOLD = 0.43   # 0805_three_model_workflow.md：三模型測試固定使用 0.43


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
        # 依 0805_three_model_workflow.md 使用 0803 版；repo 裡還沒有 0803 時退回 0802
        lstm_ver = next(v for v in ("0803", "0802")
                        if os.path.exists(os.path.join(BASE_DIR, f"{v}_lstm_model.keras")))
        lstm_model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, f"{lstm_ver}_lstm_model.keras"), compile=False)
        lstm_threshold = LSTM_THRESHOLD   # 文件規定固定 0.43，不讀 config 檔
        _models["lstm"] = {"model": lstm_model, "threshold": lstm_threshold,
                           "file": f"{lstm_ver}_lstm_model.keras"}
        logger.info(f"LSTM {lstm_ver} loaded (threshold={lstm_threshold})")
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


_SCHEME_RE = re.compile(r"^\s*[a-z][a-z0-9+.-]*://", re.IGNORECASE)   # http:// https:// ftp:// … 都拿掉（同 n8n）
_WWW_RE = re.compile(r"^www\d*\.", re.IGNORECASE)   # 和 n8n 一樣：www. / www1. / www2. 都拿掉


def _to_domain(url: str) -> str:
    """把網址縮成主機名稱再送進模型，和 n8n 流程一致：
    https://www.youtube.com/watch?v=abc → youtube.com
    1. 拿掉任何協定（http:// https:// ftp:// …）
    2. 拿掉路徑、查詢字串、#片段（第一個 / ? # 之後全部丟掉）
    3. 拿掉帳號密碼（user@）與 port（:8080）
    4. 轉小寫、拿掉結尾的點與開頭的 www.（含 www1. 這類）
    其他子網域保留（paypal.com.evil.xyz 這類冒用手法要靠它判斷）。
    """
    s = _SCHEME_RE.sub("", url.strip()).strip()
    s = re.split(r"[/?#\\]", s, maxsplit=1)[0]
    s = s.rsplit("@", 1)[-1]
    if s.startswith("["):                  # IPv6：[::1]:8080
        s = s.split("]", 1)[0] + "]"
    else:
        s = s.split(":", 1)[0]
    s = s.lower().rstrip(".")
    return _WWW_RE.sub("", s)


def _clean_for_xgb(bare: str):
    """文件 5.1：[.] (.) 還原成 .、轉小寫、去掉非 ASCII；空字串、.、.. 視為無效（回傳 None）"""
    s = bare.replace("[.]", ".").replace("(.)", ".").strip().lower()
    s = s.encode("ascii", "ignore").decode("ascii")
    return None if s in {"", ".", ".."} else s


def _clean_for_lstm(bare: str):
    """文件 5.2：轉小寫、去掉非 ASCII；空字串、.、.. 視為無效"""
    s = bare.strip().lower().encode("ascii", "ignore").decode("ascii")
    return None if s in {"", ".", ".."} else s


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
    model_input: str = ""      # 實際送進模型的字串（網域），方便和 n8n 對照


# ──────────────────────────────────────────────
# Core prediction logic
# ──────────────────────────────────────────────
def _predict_one(url: str) -> PredictionResult:
    bare = _to_domain(url)             # 送進模型的只有網域（見 _to_domain）；回傳時仍用原始 url
    if not bare:
        raise HTTPException(status_code=422, detail="url must not be empty")
    scores = {}
    votes = []
    reasons = []

    # ── XGBoost ──────────────────────────────
    if "xgb" in _models:
        try:
            import pandas as pd
            xgb_in = _clean_for_xgb(bare)
            if xgb_in is None:
                raise ValueError("網址清理後為空，XGBoost 不判斷")
            feats = extract_url_features(xgb_in, force_https=False)
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
            if clean is None:
                raise ValueError("網址清理後為空，LSTM 不判斷")
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
            # 文件 5.3：CNN 只去協定，不轉小寫、不去非 ASCII（網域在 _to_domain 已是小寫）
            ids = [char2idx.get(c, 0) for c in bare[:CNN_MAX_LEN]]
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
        model_input=bare,
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
        "lstm_file": _models.get("lstm", {}).get("file"),
    }


@app.post("/predict", response_model=PredictionResult)
def predict(item: URLItem):
    return _predict_one(item.url)


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(req: BatchRequest):
    return [_predict_one(u) for u in req.urls]
