# MNIST ETAM 關聯記憶

本專案依 `spec.md` 與 Chapter 6 的 ETAM Eq.(7–12)，實作以 MNIST 為資料的
Error Tolerant Associative Memory。它不是一般 CNN：每個 neuron 都是一個
hyperplane，訓練會移動 threshold 並旋轉法向量，讓正負 training patterns
之間的最小 margin 盡可能增加。

## 模型

- **Part I**：784 個影像 neuron，僅連接自身周圍的 5×5 window。邊界外連線
  直接刪除，保留 `wii`，且不強制 `wij = wji`。
- **Part II**：採 leakage-free 兩階段設計。十個 normalized Eq.(7)
  image-to-class hyperplanes 先進行 multiclass competition，勝出類別再展開為
  固定 96-bit template，最後由獨立的 96×96 ETAM associative memory cleanup。
  訓練 classifier 時不提供正確 template，因此測試時不會突然失去 label 訊號。
- MNIST 像素以 `>=128` 轉為 `+1`，其餘為 `-1`。測試 Part II 時，96 個
  template 初值依規格設為 `0`。
- recall 採同步更新，會區分 stable state、limit cycle 與超過步數上限。

## 安裝

建議使用 `uv` 建立隔離環境：

```bash
uv sync --extra dev
```

所有命令可寫成 `uv run etam-mnist ...`。

## 執行

下載並驗證 MNIST：

```bash
uv run etam-mnist download
```

快速模式會使用固定 seed 的 500/100 stratified split：

```bash
uv run etam-mnist train --config configs/quick.toml --part all
uv run etam-mnist evaluate --config configs/quick.toml
```

完整模式使用 MNIST 原始 training set 的 55,000/5,000 split：

```bash
uv run etam-mnist train --config configs/full.toml --part all --resume
uv run etam-mnist evaluate --config configs/full.toml
```

`--resume` 會略過 checkpoint 中已完成的 neuron。Checkpoint 同時保存
connectivity hash、split hash、訓練設定及逐 neuron 結果，避免誤用不同資料。

執行 baseline 與 4 rotations × 2 scales augmentation 比較：

```bash
uv run etam-mnist experiment --config configs/quick.toml --augmentation --resume
```

## 輸出

每個 artifact directory 會包含：

- `part1.npz`：Part I 模型與續跑資訊
- `part2_classifier.npz`：無 label leakage 的 784→10 class readout
- `part2_cleanup.npz`：96×96 template associative memory
- `part1_training.csv`、`part2_cleanup_training.csv`：逐 neuron 訓練結果
- `metrics.json`：完整評估數值
- `part1_metrics.csv`、`part1_noise.png`
- `part2_confusion.csv`、`part2_confusion.png`
- `report.md`：繁體中文實驗摘要

Part I 評估 0%、5%、10%、20%、30% bit-flip；Part II 輸出 accuracy、
per-class accuracy、confusion matrix、收斂率、limit-cycle rate 與平均步數。

## Part II 修正說明

最初依權重總數直接將正確的 96-bit template 與影像一起送入 Part II。該設計
會讓 template neurons 在訓練時直接讀到答案，但測試時 template 初值又全部為
零，造成 label leakage 與 train/test distribution shift。舊 full model 的
template-to-template 權重能量平均達 43.8%，accuracy 只有 31.04%。

目前模型將分類和 associative cleanup 分開。完整 55k/5k 結果的 Part II
accuracy 為 69.98%，舊的 `part2.npz` 僅保留作 legacy 對照，評估不再使用。

## 完整模式驗收

`configs/full.toml` 啟用 `strict_stability = true`。Part I 會檢查全部
training patterns；Part II 的十類 readout 是 multiclass competition，不要求
十個 one-vs-all 問題各自線性可分，但 96×96 cleanup memory 必須完整保存十個
templates。

執行測試：

```bash
uv run pytest
```

## 實作備註

原論文 Chapter 6 Fig.7 的十個 8×12 patterns 實際為 A–J。本專案依 MNIST
任務改採版本控制內固定的 0–9 templates。ETAM learning rate 使用參考
MATLAB/C 程式的 `alpha=0.005`；當 Eq.(12) 的候選更新不能再增加最小 margin
時，候選不會被接受，相當於復原最後一次更新。
