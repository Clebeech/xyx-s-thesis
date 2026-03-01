# -*- coding: utf-8 -*-
"""
基于 qlib Alpha360 特征 + LightGBM 预测 next return 的 pipeline。

用法:
    cd /root/thesis/pipeline
    python run.py

依赖: 已安装 qlib、lightgbm；数据路径在 config.QLIB_DATA_URI。
"""
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha360
from qlib.contrib.model.gbdt import LGBModel

from config import (
    QLIB_DATA_URI,
    REGION,
    MARKET,
    SEGMENTS,
    START_TIME,
    END_TIME,
    FIT_START_TIME,
    FIT_END_TIME,
    LGB_PARAMS,
    OUTPUT_DIR,
    SAVE_MODEL,
    SAVE_PREDICTION,
)


def main():
    # 1. 初始化 qlib
    print(f"[1/5] 初始化 qlib，数据路径: {QLIB_DATA_URI}")
    qlib.init(provider_uri=QLIB_DATA_URI, region=REG_CN)

    # 2. 构建 Alpha360 数据配置（label = next return: Ref(close,-2)/Ref(close,-1)-1）
    data_handler_config = {
        "start_time": START_TIME,
        "end_time": END_TIME,
        "fit_start_time": FIT_START_TIME,
        "fit_end_time": FIT_END_TIME,
        "instruments": MARKET,
        "infer_processors": [],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    }

    task_dataset = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": list(SEGMENTS["train"]),
                "valid": list(SEGMENTS["valid"]),
                "test": list(SEGMENTS["test"]),
            },
        },
    }

    print("[2/5] 构建 Alpha360 数据集 ...")
    dataset: DatasetH = init_instance_by_config(task_dataset, accept_types=DatasetH)

    # 3. 训练 LightGBM
    print("[3/5] 训练 LightGBM ...")
    model = LGBModel(**LGB_PARAMS)
    model.fit(dataset)

    # 4. 在测试集上预测
    print("[4/5] 在测试集上预测 ...")
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    print(f"预测结果 shape: {pred.shape}")
    print(pred.head())

    # 5. 可选：保存模型与预测
    if SAVE_MODEL or SAVE_PREDICTION:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_MODEL:
        path = OUTPUT_DIR / "lgb_alpha360_next_return.txt"
        model.model.save_model(str(path))
        print(f"模型已保存: {path}")
    if SAVE_PREDICTION:
        path = OUTPUT_DIR / "pred_test.csv"
        pred.to_csv(path)
        print(f"测试集预测已保存: {path}")

    # 简单评估：与 label 的 Spearman 相关系数（Rank IC）
    df = dataset.prepare("test", col_set=["label"], data_key=DataHandlerLP.DK_L)
    if not df.empty:
        align_idx = pred.index.intersection(df.index)
        if len(align_idx) > 0:
            try:
                from scipy.stats import spearmanr
                y_test = df.loc[align_idx]
                if isinstance(y_test.columns, pd.MultiIndex):
                    y_test = y_test.droplevel(axis=1, level=0).iloc[:, 0]
                else:
                    y_test = y_test.iloc[:, 0]
                ic, _ = spearmanr(pred.loc[align_idx].values.flatten(), y_test.values.flatten())
                print(f"[5/5] 测试集 Spearman IC: {ic:.4f}")
            except Exception as e:
                print(f"[5/5] IC 计算跳过: {e}")
        else:
            print("[5/5] 无法对齐 label，跳过 IC 计算。")
    else:
        print("[5/5] 完成。")

    return model, dataset, pred


if __name__ == "__main__":
    main()
