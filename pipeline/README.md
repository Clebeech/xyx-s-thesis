# Alpha360 + LightGBM 预测 Next Return Pipeline

基于 qlib 的 **Alpha360** 特征与 **LightGBM**，实现「预测下一日收益率」(next return) 的完整 pipeline，代码位于 `thesis/pipeline`。

## 内容说明

- **Alpha360**：过去 60 个交易日的 close / open / high / low / vwap / volume 归一化序列（共 360 维）。
- **标签**：`Ref($close, -2) / Ref($close, -1) - 1`，即 T+1 日相对 T 日的收益率。
- **模型**：LightGBM 回归（MSE），带 early stopping 与验证集。

## 目录结构

```
thesis/pipeline/
├── config.py   # 数据路径、时间区间、标的、LightGBM 超参、输出开关
├── run.py      # 主流程：qlib 初始化 → Alpha360 数据 → 训练 → 预测 → 保存与 IC
├── output/     # 运行后生成：模型、测试集预测
└── README.md   # 本说明
```

## 依赖

- `qlib`
- `lightgbm`
- `pandas`, `numpy`
- `scipy`（用于 Spearman IC）

数据需已按 qlib 格式存放在 `config.QLIB_DATA_URI`（默认：`/root/autodl-tmp/data/.qlib/qlib_data/cn_data`）。

## 运行

```bash
cd /root/thesis/pipeline
python run.py
```

流程包括：

1. 使用 `config.QLIB_DATA_URI` 初始化 qlib  
2. 构建 Alpha360 数据集（train/valid/test）  
3. 训练 LightGBM  
4. 在测试集上预测并打印 Spearman IC  
5. 若 `config.SAVE_MODEL` / `SAVE_PREDICTION` 为 True，将模型与预测写入 `output/`

## 配置

在 `config.py` 中可修改：

- **QLIB_DATA_URI**：qlib 数据根目录  
- **MARKET**：标的池（如 `csi300`、`csi500`）  
- **START_TIME / END_TIME / SEGMENTS**：数据时间范围与 train/valid/test 划分  
- **LGB_PARAMS**：LightGBM 超参数  
- **SAVE_MODEL / SAVE_PREDICTION**：是否保存模型与预测结果  

## 输出

- `output/lgb_alpha360_next_return.txt`：LightGBM 模型（当 `SAVE_MODEL=True`）  
- `output/pred_test.csv`：测试集预测分数（当 `SAVE_PREDICTION=True`）  
- 终端打印测试集 **Spearman IC**（预测与真实 next return 的秩相关）
