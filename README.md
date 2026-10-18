# PhishGuard AI — 釣魚網址偵測 API

以 **XGBoost + LSTM + CharCNN** 三個模型集成判斷網址是否為釣魚網站，用 FastAPI 提供 HTTP API，部署在 Render。

- 線上網址：`https://n8n-local-model-university.onrender.com`
- 目前模型版本：**0802**
- 使用者：PhishGuard 網站（`server.py` 轉發）、Chrome 擴充功能、n8n workflow

---

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `main.py` | FastAPI 主程式：啟動時載入模型、提供 `/health`、`/predict`、`/predict/batch` |
| `transfer.py` | XGBoost 用的網址特徵擷取（UF2–UF26） |
| `0802_augmented_xgb_model.ubj` | XGBoost 模型（XGBoost 原生 UBJSON 格式） |
| `0802_augmented_model_config.pkl` | XGBoost 門檻：`{"best_threshold": 0.26}` |
| `0802_lstm_model.keras` | LSTM 模型（Keras 3，模型內含 `TextVectorization`，輸入是字串） |
| `0802_lstm_config.json` | LSTM 門檻：`{"threshold": 0.43}` |
| `cnn_model.onnx` | CharCNN 模型（**Render 實際載入的是這個**） |
| `char_cnn_model.pth` | CharCNN 的 PyTorch 權重，只用來轉成 ONNX，Render 不會讀 |
| `tokenizer_CNN.pkl` | CharCNN 字元表（125 個字元，`<PAD>` = 0） |
| `convert_cnn_to_onnx.py` | 把 `char_cnn_model.pth` 轉成 `cnn_model.onnx`，並驗證兩者輸出一致 |
| `requirements.txt` | Python 套件（不含 PyTorch，CharCNN 用 ONNX Runtime 執行） |
| `Dockerfile` | Python 3.11-slim 映像檔 |
| `render.yaml` | Render 部署設定（Docker、free 方案、health check `/health`） |

---

## 判斷流程

```
輸入網址
  │  縮成網域：去掉 http(s)://、路徑、查詢字串、port、開頭的 www.
  │  例：https://www.youtube.com/watch?v=abc → youtube.com（與 n8n 流程一致）
  ▼
┌──────────────┬──────────────────┬──────────────────┐
│ XGBoost      │ LSTM             │ CharCNN (ONNX)   │
│ 23 個網址特徵 │ 轉小寫、只留 ASCII │ 轉小寫           │
│ (transfer.py)│ 直接餵字串        │ 字元表編碼、長度 200│
└──────┬───────┴────────┬─────────┴────────┬─────────┘
       ▼                ▼                  ▼
   釣魚機率 0~1      釣魚機率 0~1        釣魚機率 0~1
       └────────────────┼──────────────────┘
                        ▼
              平均機率 → verdict
              多數決   → is_phishing
```

### 各模型前處理

| 模型 | 輸入處理 | 門檻 |
|---|---|---|
| XGBoost | 縮成網域後交給 `extract_url_features()`，取 23 個特徵：UF2、UF3、UF4、UF5、UF6、UF8、UF10–UF26（不含 UF1、UF7、UF9），用 DataFrame 帶欄位名稱 | 0.26 |
| LSTM | 縮成網域、去掉非 ASCII 字元，以字串送入模型（模型內的 `TextVectorization` 逐字元切分，長度 200） | 0.43 |
| CharCNN | 縮成網域，用 `tokenizer_CNN.pkl` 轉成索引（找不到的字元 = 0），截斷或補 0 到長度 200 | 0.40 |

### CharCNN 結構

```
Embedding(125, 64, padding_idx=0)
Conv1d(64→128, k=5, 無 padding) + ReLU + MaxPool1d(2)
Conv1d(128→128, k=3, 無 padding) + ReLU + MaxPool1d(2)
Conv1d(128→64,  k=3, 無 padding) + ReLU + AdaptiveMaxPool1d(1)
Linear(64, 64) + ReLU + Dropout(0.3)
Linear(64, 1) + Sigmoid
```

### 集成規則

- `confidence` = 三個模型釣魚機率的**平均**
- `verdict`：平均 ≥ 0.65 → `phishing`；≥ 0.40 → `suspicious`；其餘 → `safe`
- `is_phishing`：各模型與自己的門檻比較後**多數決**
- 某個模型載入或預測失敗時只會記 log 並略過，用剩下的模型繼續算（所以部署後一定要看 `/health`）

### `reasons`（判斷依據）

由 XGBoost 的特徵以固定規則產生，**不是** SHAP 或模型解釋，只代表網址具有某個特徵：

| 特徵 | 說明 |
|---|---|
| UF3 | 使用 IP 位址作為主機名稱 |
| UF4 | URL 含有 @ 符號 |
| UF10 | 使用短網址服務 |
| UF11 | 主機名稱含有連字號 (-) |
| UF12 | 含有敏感關鍵字（login / verify / bank…） |
| UF13 | 冒用知名品牌名稱 |
| UF18 | 含有 URL 編碼字元 (%) |
| UF20 | 品牌名出現在子網域（非主網域） |
| UF24 | 使用免費虛擬主機服務 |
| UF26 | 使用高風險頂級域名（.sbs / .xyz…） |

---

## API

### `GET /health`

```json
{
  "status": "ok",
  "model_version": "0802",
  "models_loaded": ["xgb", "lstm", "cnn"],
  "thresholds": { "xgb": 0.26, "lstm": 0.43, "cnn": 0.4 }
}
```

`models_loaded` 一定要有三個，少了哪個就代表那個模型載入失敗。

### `POST /predict`

```bash
curl -X POST https://n8n-local-model-university.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://paypa1-secure-login.verify-account.xyz/signin.php"}'
```

```json
{
  "url": "https://paypa1-secure-login.verify-account.xyz/signin.php",
  "is_phishing": true,
  "confidence": 0.9999,
  "verdict": "phishing",
  "scores": { "xgboost": 0.9999, "lstm": 0.9999, "cnn": 1.0 },
  "reasons": ["主機名稱含有連字號 (-)", "使用高風險頂級域名（.sbs / .xyz…）", "含有敏感關鍵字（login / verify / bank…）"],
  "model_input": "paypa1-secure-login.verify-account.xyz"
}
```

| 欄位 | 說明 |
|---|---|
| `url` | 使用者原本輸入的網址（含協定也照原樣回傳） |
| `is_phishing` | 多數決結果 |
| `confidence` | 平均釣魚機率，0~1，越高越可能是釣魚 |
| `verdict` | `safe` / `suspicious` / `phishing` |
| `scores` | 各模型的釣魚機率 |
| `reasons` | 命中的規則特徵 |
| `model_input` | 實際送進模型的字串（網域），用來和 n8n 對照 |

錯誤：網址是空的或只有 `https://` → `422`；三個模型都不能用 → `503`。

### `POST /predict/batch`

```json
{ "urls": ["https://www.google.com", "http://paypa1.xyz"] }
```

一次最多 **20** 個，回傳陣列，每一筆格式同 `/predict`。

---

## 本機執行

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

或用 Docker（和 Render 相同環境）：

```bash
docker build -t phishguard-api .
docker run -p 8001:8000 phishguard-api
```

---

## 部署（Render）

Render 服務連結這個 repo 的 `main` branch，**push 之後會自動重新部署**（Docker 建置約數分鐘）。沒有自動開始時，到 Render 服務頁按 **Manual Deploy → Deploy latest commit**。

- 免費方案閒置約 15 分鐘會休眠，休眠後第一次呼叫要等 30 秒以上喚醒。
- 部署失敗或模型有問題時，到服務的 **Events** 找上一次成功的部署按 **Rollback**。
- 實測記憶體用量約 900 MB（TensorFlow 佔大部分）。如果 Logs 出現 `Ran out of memory`，需要升級方案。

---

## 更新模型

1. **XGBoost**：放入新的 `.ubj` 與門檻 `.pkl`。特徵必須仍是 `transfer.py` 產生的同樣 23 個、同樣順序，否則要一起改 `XGB_FEATURE_COLS` 與 `transfer.py`。
2. **LSTM**：放入新的 `.keras` 與門檻 `.json`。模型第一層必須是 `TextVectorization`（字串輸入）；如果改成整數輸入，`main.py` 的 LSTM 預測段要跟著改。
3. **CharCNN**：放入新的 `char_cnn_model.pth`（字元表有變就一起換 `tokenizer_CNN.pkl`），在有 PyTorch 的環境執行：
   ```bash
   pip install torch onnx onnxscript onnxruntime
   python convert_cnn_to_onnx.py
   ```
   腳本會產生 `cnn_model.onnx` 並檢查與 PyTorch 輸出一致。**一定要上傳 `cnn_model.onnx`**，只換 `.pth` 不會生效。結構有改時，要先改 `convert_cnn_to_onnx.py` 裡的 `CharCNN`。
4. 檔名有變時，修改 `main.py` 裡 `lifespan()` 的檔名，並更新 `MODEL_VERSION`。
5. **所有檔案在同一個 commit 上傳**，避免 Render 部署到一半的版本。
6. 部署後檢查：
   ```bash
   curl https://n8n-local-model-university.onrender.com/health
   ```
   確認 `model_version` 正確、`models_loaded` 有三個，再用 `/predict` 測幾個已知的正常與釣魚網址，`scores` 裡三個分數都要出現。

---

## 更新紀錄

### 0802-domain（2026-10-18）

- 送進模型前把網址縮成網域（去掉協定、路徑、查詢字串、帳號、port、開頭的 `www.`（含 `www1.` 這類），其他子網域保留），與 n8n 流程一致。
  原因：LSTM / CNN 對「`www.` + 路徑」的正常網址誤判嚴重（例如 `www.youtube.com/watch?v=…` LSTM 0.997）。
- 代價：只看網域後，路徑上的線索（`/login`、`@`、`%` 編碼）不再列入判斷；冒用子網域、IP 網址的分數也會變低。
- 回應新增 `model_input` 欄位。

### 0802（2026-10-17）

- 更新三個模型：XGBoost 0802（`.ubj`，門檻 0.35 → 0.26）、LSTM 0802（門檻 0.43）、CharCNN hdscan 版本。
- **修正 LSTM 從未生效**：模型輸入是字串，舊版程式送整數編碼導致每次預測都失敗而被略過。
- **修正 CharCNN 結構錯誤**：舊版 ONNX 用了錯誤的結構（有 padding、每層全域池化），輸出幾乎固定在 0.3–0.5；改為與訓練一致的結構並重新轉檔。
- 三個模型統一先去掉 `http://` / `https://` 再預測，與訓練資料一致。
- `/health` 新增 `model_version` 與各模型門檻。
- 移除程式中未使用的 PyTorch 模型定義；新增 `.gitignore`。

### 0623

- 初版：XGBoost + LSTM + CharCNN 集成 API。
