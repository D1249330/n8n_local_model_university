"""
執行一次就好：把 char_cnn_model.pth 轉成 cnn_model.onnx
用法：python convert_cnn_to_onnx.py
需要先安裝：pip install torch onnx
"""
import os
import pickle
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(__file__)
CNN_MAX_LEN = 200
CNN_EMB_DIM = 64

# ── 定義模型（與 main.py 相同架構）──────────────────
class CharCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, CNN_EMB_DIM, padding_idx=0)
        self.conv1 = nn.Conv1d(CNN_EMB_DIM, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64, 64)
        self.fc2   = nn.Linear(64, 1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        e = self.embedding(x).permute(0, 2, 1)
        e = self.relu(self.conv1(e)).max(dim=2)[0].unsqueeze(2)
        e = self.relu(self.conv2(e)).max(dim=2)[0].unsqueeze(2)
        e = self.relu(self.conv3(e)).max(dim=2)[0]
        e = self.relu(self.fc1(e))
        return torch.sigmoid(self.fc2(e)).squeeze(1)

# ── 載入 tokenizer & 權重 ────────────────────────────
with open(os.path.join(BASE_DIR, "tokenizer_CNN.pkl"), "rb") as f:
    char2idx = pickle.load(f)

vocab_size = len(char2idx)   # 160，不需要 +1
model = CharCNN(vocab_size)
state = torch.load(os.path.join(BASE_DIR, "char_cnn_model.pth"), map_location="cpu")
model.load_state_dict(state)
model.eval()

# ── 輸出 ONNX ────────────────────────────────────────
dummy_input = torch.zeros(1, CNN_MAX_LEN, dtype=torch.long)
output_path = os.path.join(BASE_DIR, "cnn_model.onnx")

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)

print(f"✅ 轉換完成：{output_path}")
print(f"   檔案大小：{os.path.getsize(output_path) / 1024:.1f} KB")
