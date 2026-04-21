#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/12/23 16:20
# @File  : create_question_real_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 从循证问题数据生成训练用的 parquet 文件（RL训练只需要 question 字段）

import json
import uuid
import pandas as pd
import os
import logging

OUTPUT_DIR = "data"
TRAIN_FILE = "train.parquet"
DEV_FILE = "val.parquet"

# 循证问题数据文件路径（RL训练只需要 question，不需要 answer）
RAW_FILE = "./evidence_questions_1000.jsonl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealDataBuilder")


def load_real_questions(jsonl_path: str):
    """
    从 evidence_questions_1000.jsonl 里读取循证问题

    每行格式：
    {"id": "...", "question": "...", "answer": "...", "specialty": "...", ...}

    注意：RL训练只需要 question 字段，answer 由模型动态生成，奖励由 DeepSeek 教师模型评分
    """
    data = []
    skipped_question_empty = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"跳过无法解析的行 {line_idx}: {line[:100]} ...")
                continue

            # 验证 question 字段（唯一必需字段）
            question = obj.get("question", "").strip() if obj.get("question") else ""
            if not question:
                skipped_question_empty += 1
                continue

            # 使用原始 id 或生成新 id
            sample_id = obj.get("id", "") or f"evq_{uuid.uuid4().hex[:8]}"

            data.append({
                "id": sample_id,
                "question": question,
            })

    logger.info(f"✅ 从 {jsonl_path} 读取到 {len(data)} 条有效问题")
    if skipped_question_empty > 0:
        logger.warning(f"⚠️ 跳过 {skipped_question_empty} 条空 question 记录")

    return data


def build_and_save_dataset():
    # 1. 加载循证问题
    data_list = load_real_questions(RAW_FILE)
    if not data_list:
        logger.warning("没有有效数据，终止。")
        return

    # 2. 转成 DataFrame，只保留 id 和 question
    df = pd.DataFrame(data_list)
    df = df[["id", "question"]]

    logger.info(f"📊 数据字段: {list(df.columns)}")
    logger.info(f"📊 数据样例:\n{df.head()}")

    # 3. Train / Val 划分（9:1）
    if len(df) > 10:
        df_val = df.sample(frac=0.1, random_state=42)
        df_train = df.drop(df_val.index)
    else:
        df_train = df
        df_val = df.iloc[0:0]

    # 4. 保存
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    train_path = os.path.join(OUTPUT_DIR, TRAIN_FILE)
    val_path = os.path.join(OUTPUT_DIR, DEV_FILE)

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    logger.info(f"✅ 保存训练集: {len(df_train)} 条 -> {train_path}")
    logger.info(f"✅ 保存验证集: {len(df_val)} 条 -> {val_path}")


if __name__ == "__main__":
    build_and_save_dataset()
