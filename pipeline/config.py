# -*- coding: utf-8 -*-
"""
Alpha360 + LightGBM 预测 next return 的 pipeline 配置。
数据路径、时间区间、标的、模型超参等均可在此修改。
"""
from pathlib import Path

# ---------- 数据 ----------
# qlib 数据目录（需包含 calendars / features / instruments）
QLIB_DATA_URI = "/root/autodl-tmp/data/.qlib/qlib_data/cn_data"
REGION = "cn"

# 标的池
MARKET = "csi300"
BENCHMARK = "SH000300"

# 全量数据时间范围
START_TIME = "2008-01-01"
END_TIME = "2020-08-01"
# 用于 learn_processors 拟合的时间范围（如 ZScoreNorm）
FIT_START_TIME = "2008-01-01"
FIT_END_TIME = "2014-12-31"

# 训练 / 验证 / 测试 划分
SEGMENTS = {
    "train": ("2008-01-01", "2014-12-31"),
    "valid": ("2015-01-01", "2016-12-31"),
    "test": ("2017-01-01", "2020-08-01"),
}

# ---------- 模型 ----------
# LightGBM 超参数（与 qlib 官方 Alpha360 benchmark 一致，可按需改）
LGB_PARAMS = {
    "loss": "mse",
    "num_threads": 20,
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
    "verbose_eval": 20,
    "colsample_bytree": 0.8879,
    "learning_rate": 0.0421,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
}

# ---------- 输出 ----------
# 模型与预测结果保存目录（None 表示不保存到磁盘）
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SAVE_MODEL = True
SAVE_PREDICTION = True
