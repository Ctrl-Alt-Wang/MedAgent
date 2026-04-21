from datasets import Dataset

data = {
    "question": [
        "查询昨天新增用户数",
        "统计订单总金额",
        "按城市分组统计用户数量"
    ]
}

dataset = Dataset.from_dict(data)

dataset.to_parquet("sql_agent_questions.parquet")