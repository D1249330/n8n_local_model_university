"""
把 char_cnn_model.pth 轉成 cnn_model.onnx（Render 上只載入 .onnx，不需要 PyTorch）
用法：python convert_cnn_to_onnx.py
需要先安裝：pip install torch onnx onnxscript onnxruntime

模型結構依 0805_three_model_test.py（見 cnn_config 註記）：
  Embedding(vocab, 64, padding_idx=0)
  Conv1d(64→128, k=5, 無 padding) + ReLU + MaxPool1d(2)
  Conv1d(128→128, k=3, 無 padding) + ReLU + MaxPool1d(2)
  Conv1d(128→64,  k=3, 無 padding) + ReLU + AdaptiveMaxPool1d(1)
  Linear(64, 64) + ReLU + Dropout(0.3)
  Linear(64, 1) + Sigmoid
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_MAX_LEN = 200
CNN_EMB_DIM = 64


class CharCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, CNN_EMB_DIM, padding_idx=0)
        self.conv1 = nn.Conv1d(CNN_EMB_DIM, 128, kernel_size=5)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        e = self.embedding(x).permute(0, 2, 1)          # (B, 64, L)
        e = self.pool(self.relu(self.conv1(e)))
        e = self.pool(self.relu(self.conv2(e)))
        e = self.global_pool(self.relu(self.conv3(e))).squeeze(2)
        e = self.dropout(self.relu(self.fc1(e)))
        return torch.sigmoid(self.fc2(e)).squeeze(1)


with open(os.path.join(BASE_DIR, "tokenizer_CNN.pkl"), "rb") as f:
    char2idx = pickle.load(f)

model = CharCNN(len(char2idx))
state = torch.load(os.path.join(BASE_DIR, "char_cnn_model.pth"), map_location="cpu")
model.load_state_dict(state)
model.eval()

dummy_input = torch.zeros(1, CNN_MAX_LEN, dtype=torch.long)
output_path = os.path.join(BASE_DIR, "cnn_model.onnx")

export_kwargs = dict(
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
try:
    torch.onnx.export(model, dummy_input, output_path, dynamo=False, **export_kwargs)
except TypeError:  # 舊版 torch 沒有 dynamo 參數
    torch.onnx.export(model, dummy_input, output_path, **export_kwargs)

# 驗證：ONNX 和 PyTorch 的輸出要一致
import onnxruntime as ort

x = torch.randint(1, len(char2idx), (4, CNN_MAX_LEN))
with torch.no_grad():
    ref = model(x).numpy()
got = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"]).run(
    None, {"input": x.numpy().astype(np.int64)})[0]
diff = float(np.abs(ref - got).max())
assert diff < 1e-4, f"ONNX 與 PyTorch 輸出不一致（最大差 {diff}）"

print(f"✅ 轉換完成：{output_path}")
print(f"   檔案大小：{os.path.getsize(output_path) / 1024:.1f} KB，與 PyTorch 最大差異 {diff:.2e}")
