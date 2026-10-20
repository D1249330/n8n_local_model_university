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
MODEL_VERSION = "0802-md-www"

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


# ── 前處理：完全依照 0805_three_model_workflow.md 第 5 節（= 0805_three_model_test.py）──
# 送進模型的是完整網址（只拿掉協定），不縮成網域。三個模型的清理方式各不相同，不能混用。

# 本專案額外規則（md 檔沒有）：開頭的 www.（含 www1.、www[.]）先拿掉，路徑與參數保留，和 n8n 一致
# https://www.youtube.com/watch?v=abc → https://youtube.com/watch?v=abc
_WWW_RE = re.compile(r"^(\s*(?:[a-zA-Z][a-zA-Z0-9+.-]*://)?)www\d*(?:\.|\[\.\]|\(\.\))", re.IGNORECASE)


def _strip_www(url: str) -> str:
    return _WWW_RE.sub(r"\1", str(url), count=1)


def _clean_url(url: str, strip_any_scheme: bool):
    """0805 的 clean_url()：XGBoost 用 strip_any_scheme=True，LSTM 用 False"""
    s = str(url).replace("[.]", ".").replace("(.)", ".").strip()
    if strip_any_scheme:
        s = re.sub(r"^[a-zA-Z]+://", "", s)
    else:
        s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
    s = s.strip().lower().encode("ascii", "ignore").decode("ascii")
    return None if s in {"", ".", ".."} else s


def _clean_url_cnn(url: str) -> str:
    """0805 的 clean_url_cnn()：只拿掉 http(s)://，不轉小寫、不還原 [.]、不去非 ASCII"""
    return re.sub(r"^https?://", "", str(url).strip(), flags=re.IGNORECASE)


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
    is_phishing: Optional[bool]      # 多數決（y_pred）；任一模型沒有結果時為 null，依文件不判定
    confidence: Optional[float]      # 投票比例 y_prob = 判為 1 的模型數 / 3（0、0.3333、0.6667、1）
    verdict: str                     # "phishing"（y_pred=1）| "safe"（y_pred=0）| "unknown"（不判定）
    scores: dict                     # 各模型原始機率 xgb_prob / lstm_prob / cnn_prob
    votes: dict = {}                 # 各模型 0/1 預測 xgb_pred / lstm_pred / cnn_pred
    reasons: list[str]               # XGBoost 特徵對應的中文說明
    model_input: dict = {}           # 各模型實際收到的字串


# ──────────────────────────────────────────────
# Core prediction logic
# ──────────────────────────────────────────────
def _predict_one(url: str) -> PredictionResult:
    if not str(url).strip():
        raise HTTPException(status_code=422, detail="url must not be empty")
    original_url = url
    url = _strip_www(url)              # 先拿掉 www.，之後完全照 md 檔前處理
    scores, votes, reasons, inputs = {}, {}, [], {}

    # ── XGBoost（文件 5.1、7.1）────────────────
    if "xgb" in _models:
        try:
            import pandas as pd
            xgb_in = _clean_url(url, strip_any_scheme=True)
            if xgb_in is not None:
                inputs["xgb"] = xgb_in
                feats = extract_url_features(xgb_in, force_https=False)
                x = pd.DataFrame([[feats.get(c, 0) for c in XGB_FEATURE_COLS]], columns=XGB_FEATURE_COLS)
                prob = float(_models["xgb"]["model"].predict_proba(x)[0][1])
                scores["xgb"] = round(prob, 4)
                votes["xgb"] = int(prob >= _models["xgb"]["threshold"])
                _add_reasons(feats, reasons)
        except Exception as e:
            logger.warning(f"XGB predict error: {e}")

    # ── LSTM（文件 5.2、7.2）───────────────────
    if "lstm" in _models:
        try:
            import tensorflow as tf
            lstm_in = _clean_url(url, strip_any_scheme=False)
            if lstm_in is not None:
                inputs["lstm"] = lstm_in
                prob = float(_models["lstm"]["model"].predict(tf.constant([[lstm_in]], dtype=tf.string), verbose=0)[0][0])
                scores["lstm"] = round(prob, 4)
                votes["lstm"] = int(prob >= _models["lstm"]["threshold"])
        except Exception as e:
            logger.warning(f"LSTM predict error: {e}")

    # ── CharCNN（文件 5.3、7.3）────────────────
    if "cnn" in _models:
        try:
            cnn_in = _clean_url_cnn(url)
            inputs["cnn"] = cnn_in
            char2idx = _models["cnn"]["char2idx"]
            ids = [char2idx.get(c, 0) for c in cnn_in][:CNN_MAX_LEN]
            ids = ids + [0] * (CNN_MAX_LEN - len(ids))
            out = _models["cnn"]["session"].run(None, {"input": np.array([ids], dtype=np.int64)})
            prob = float(out[0][0])
            scores["cnn"] = round(prob, 4)
            votes["cnn"] = int(prob >= _models["cnn"]["threshold"])
        except Exception as e:
            logger.warning(f"CNN predict error: {e}")

    # ── 三模型硬投票（文件第 8 節）──────────────
    # 任一模型沒有有效預測 → 依文件不判定（NaN）
    if len(votes) < 3:
        return PredictionResult(url=original_url, is_phishing=None, confidence=None, verdict="unknown",
                                scores=scores, votes=votes, reasons=sorted(set(reasons)), model_input=inputs)
    total = sum(votes.values())
    y_pred = 1 if total >= 2 else 0
    return PredictionResult(
        url=original_url,
        is_phishing=bool(y_pred),
        confidence=round(total / 3.0, 4),
        verdict="phishing" if y_pred else "safe",
        scores=scores,
        votes=votes,
        reasons=sorted(set(reasons)),
        model_input=inputs,
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
