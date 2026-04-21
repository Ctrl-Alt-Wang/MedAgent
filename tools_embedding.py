#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/10/29
# @File  : adk_tools.py
# @Contact : github: johnson7788
# @Desc  : 基于向量搜索，最新搜索

import logging
import os
import json
import time
import re
import asyncio
from typing import List, Dict, Any, Union, Optional
from agents import function_tool
from datetime import datetime
import uuid


    
# from rapidfuzz import fuzz
import dotenv

# 用于异步HTTP请求
import aiohttp

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# module_path = os.path.dirname(os.path.abspath(__file__))
# module_log_file = os.path.join(module_path, "embedding_tools.log")
# file_handler = logging.FileHandler(module_log_file, encoding="utf-8")
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter(
#     '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


DB_MAPPER = {
    "search_guideline_db": "search_guideline_db",
    "search_clinical_db": "search_clinical_db",
    "search_meta_db": "search_meta_db",
    "search_guidelinezh_db": "search_guidelinezh_db",
}

TIER_MAPPING = {
    5: "【（顶层）全球顶刊】",
    4: "【（一层）科室重磅期刊】",
    3: "【（二层）科室一定影响力期刊】",
    2: "【（三层）补充期刊】",
    1: "【其他期刊】"
}

key_mapping = {
    "search_guidelinezh_db": "中文指南",
    "search_guideline_db": "英文指南",
    "search_meta_db": "系统评价和meta分析",
    "search_clinical_db": "RCT"
}
# ============================================

def fuzzy_search(keyword: str, content: str, idprefix="01", db_id="01") -> Dict[str, Any]:
    """模糊搜索匹配函数 (保持同步，因为是纯计算任务)"""
    if content is None:
        content = ""

    # 去除HTML标签
    content = re.sub(r'<[^>]+>', '', content).strip()
    
    # 截断过长文档，防止第二轮 prompt 超过 max_model_len
    if len(content) > 1500:
        content = content[:1500] + "...(内容已截断)"
    # 不切割，直接是一整句
    sentences = [content] if content else []

    if not sentences:
        return {
            "match_sentence": "",
            "match_content": "",
            "match_sentences": [
                {"id": f"{idprefix}-0", "sentence": "", "db_id": db_id, "prefix_sentence": "", "tail_sentence": ""}]
        }

    # 全文即所有的内容，不再进行相似度切割匹配
    match_content = sentences[0]

    # 构建结果
    match_sentences = []
    # 使用 uuid 确保在高并发或快速循环下的唯一性
    unique_suffix = uuid.uuid4().hex[:4]
    if os.environ.get("PROJECT_NAME", "xunzheng"):
        match_id = f"{idprefix}-{unique_suffix}-0"
    else:
        match_id = f"{idprefix}_{unique_suffix}_0"

    match_sentences.append({
        "id": match_id,
        "sentence": match_content,
        "db_id": db_id,
        "prefix_sentence": "",
        "tail_sentence": ""
    })

    return {
        "match_sentence": "",
        "match_content": "",
        "match_sentences": match_sentences
    }

class MockToolContext:
    """模拟 Google ADK 的 ToolContext，用于本地测试"""
    def __init__(self):
        # 初始化 state，模拟 ADK 中的 state 字典
        self.state = {
            "search_dbs": [],
            "latest_guideline_date": None
        }
        # 模拟 user_content
        class Part:
            text = "用户提到的测试问题"
        class UserContent:
            parts = [Part()]
        self.user_content = UserContent()

async def milvus_item_search(keywords: str, 
                             collection_name: str, dense_name: str, sparse_name: str, content_chunk_field: str, output_fields: list[str], 
                             num: int, Threshold: float, db_key: str, filter_condition: str = "", filter_date_limit: str = ""):
    """
    Milvus 查询+返回权重前三的数据。
    
    :param keywords: 查询内容
    :type keywords: str
    :param tool_context: 工具上下文
    :type tool_context: ToolContext
    :param collection_name: 目标集合名称
    :type collection_name: str
    :param dense_name: 目标集合密集向量字段名称
    :type dense_name: str
    :param sparse_name: 目标集合稀疏向量字段名称    
    :type sparse_name: str
    :param content_chunk_field: 目标集合原文段落字段名称
    :type content_chunk_field: str
    :param output_fields: 需要输出的字段名称列表
    :type output_fields: list[str]
    :param num: 召回数量
    :type num: int
    :param Threshold: 阈值，低于该值的数据会被过滤
    :type Threshold: float
    :param db_key: 查询工具对应的key
    :type db_key: str
    :param filter_condition: 过滤条件
    :type filter_condition: str
    :param filter_date_limit: 过滤日期
    :type filter_date_limit: str
    """
    URL = "https://ai.infox-med.com/UserQueryProcess/GeneralQueryInterface"
    HEADERS = {"Content-Type": "application/json"}
    
    hits = []
    async with aiohttp.ClientSession() as session:
        data_body = {
            "collection_name": collection_name,
            "query": keywords,
            "dense_name": dense_name,
            "sparse_name": sparse_name,
            "content_chunk_field": content_chunk_field,
            "output_fields": output_fields,
            "filter_condition": filter_condition,
            "num": num,
            "Threshold": Threshold,
        }
        try:
            async with session.post(URL, json=data_body, headers=HEADERS) as response:
                if response.status == 200:
                    result = await response.json()
                    hits = result.get("results", [])
                    # logger.info(f"API返回 {len(hits)} 条结果")
        except Exception as e:
            logger.error(f"Milvus query failed: {e}")
            hits = []

    merged = {}

    # 如果有新结果，进行处理
    if hits:
        # 计算每个结果的权重 (tier × reranker_score)
        scored_hits = []
        for hit in hits:         
            tier = hit.get("tier", 0)
            try:
                scored_hits.append({
                    "weight": tier,
                    "hit": hit
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"计算权重失败: {e}")
                continue
        
        # 按权重降序排序
        scored_hits.sort(key=lambda x: x["weight"], reverse=True)

        # 全部文章
        top_hits = scored_hits
        
        for item_data in top_hits:
            max_weight = item_data["weight"]
            best_result = item_data["hit"]
            _id = best_result.get("doc_id", "")
            
            if not _id: 
                continue

            # 处理文档基础信息
            pdf_name = best_result.get("doc_title", "") or best_result.get("title", "")
            #发表机构
            framer = best_result.get("framer", "")
            plain_content = best_result.get("doc_content_chunk", "")
            doc_publish_time = best_result.get("doc_publish_time", "")
            # 文章摘要
            doc_abstract = best_result.get("doc_abstract", "")
            finnal_content = doc_abstract + "\n" + plain_content
            #期刊等级
            tier = best_result.get("tier", 0)
            reranker_score = best_result.get("reranker_score", 0.0)
            try:
                tier_val = int(float(tier)) if tier is not None else 1
            except (ValueError, TypeError):
                tier_val = 1
            
            tier_label = TIER_MAPPING.get(tier_val, "其他期刊")
            
            # 根据数据库类型设置不同的 idprefix
            idprefix_mapping = {
                "search_guidelinezh_db": "01",    # 中文指南
                "search_guideline_db": "02",   # 英文指南
                "search_meta_db": "03",        # 系统评价和Meta分析
                "search_clinical_db": "04"     # RCT
            }
            idprefix = idprefix_mapping.get(db_key, "00")
            
            # 模糊搜索匹配句子
            fuzzy_res = fuzzy_search(keywords, finnal_content, idprefix=idprefix, db_id=_id)

            if _id in merged:
                # 合并逻辑：若ID已存在，合并 match_sentences
                existed = merged[_id]

                # 更新权重(如果新权重更高)
                if max_weight > existed.get("weight", 0):
                    existed["weight"] = max_weight

                existing_sentences_text = set(s.get("sentence", "") for s in existed.get("match_sentences", []))
                for new_s in fuzzy_res.get("match_sentences", []):
                    # 如果这段新文字已经存在于旧的文字中，也不要拼接
                    if new_s.get("sentence") not in existing_sentences_text:
                        existed["match_sentences"].append(new_s)
                        existing_sentences_text.add(new_s.get("sentence"))

                merged[_id] = existed
            else:
                # 新ID，创建新条目
                link = f"https://www.infox-med.com/#/articleDetails?id={_id}"
                # 格式化发布日期
                publish_date = ""
                if doc_publish_time:
                    try:
                        # 尝试解析日期格式
                        dt = datetime.fromisoformat(str(doc_publish_time).replace('Z', '+00:00'))
                        publish_date = dt.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        publish_date = str(doc_publish_time)

                # 获取额外字段，如果不存在则使用 "unknown"
                framer = best_result.get("framer", "") or best_result.get("full_name_new", "")
                zky_area = best_result.get("zky_area", "")
                authors = best_result.get("authors") or best_result.get("author") or "unknown"
                journal = best_result.get("journal") or "unknown"
                doc_if = best_result.get("doc_if") or ""
                # publication_type = best_result.get("publication_type") or best_result.get("pub_type") or "unknown"

                item = {
                    "id": _id,
                    "title": pdf_name.title() if pdf_name else "",
                    "key": DB_MAPPER.get(db_key, db_key),
                    "match_sentence": "",
                    "match_sentences": fuzzy_res["match_sentences"],
                    "url": link,
                    "_matched_keywords": [keywords],
                    "publish_time": publish_date,
                    "doc_source": framer,
                    "zky_area": zky_area,
                    "docIf":doc_if,
                    "weight": max_weight,
                    "reranker_score": reranker_score,
                }
                merged[_id] = item

    # 转换回列表并按权重排序
    final_items = list(merged.values())
    # final_items.sort(key=lambda x: x.get("weight", 0), reverse=True)

    # tool_context.state["search_dbs"] = final_items

    return final_items

async def search_guideline_zh(P,I,C,O) -> Dict[str, Any]:
    """
    Asynchronous search medical Chinese guideline data.
    分4次搜索：P → PI → PIC → PICO，合并结果并按id去重（保留最高权重）

    :param P: P（人群）
    :param I: I（干预）
    :param C: C（比较）
    :param O: O（结局）
    :return: JSON object with search results
    """
    search_keywords = [
        P.strip(),                    # P
        f"{I}".strip(),           # PI
        f"{P} {I}".strip(),  # PI
        f"{P} {I} {C}".strip(),       # PIC
        f"{P} {I} {O}".strip(),  # PIO
        f"{P} {I} {C} {O}".strip(),   # PICO
    ]

    # 去重合并的结果
    merged_results = {}

    # 构建异步任务列表，并发执行所有关键词组合搜索
    tasks = []
    valid_keywords = []
    for idx, keywords in enumerate(search_keywords):
        if not keywords.strip():
            continue
        valid_keywords.append(keywords)
        tasks.append(
            milvus_item_search(keywords=keywords,
                              collection_name="guideline_zh",
                              dense_name="doc_content_chunk_dense",
                              sparse_name="doc_content_chunk_sparse",
                              content_chunk_field="doc_content_chunk",
                              output_fields=["doc_abstract","doc_title", "doc_content_chunk", "doc_publish_time", "tier","doc_id","doc_if","framer","zky_area"],
                              num=20,
                              Threshold=0.1,
                              db_key="search_guidelinezh_db")
        )

    # 并发执行所有搜索任务
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for kw, data in zip(valid_keywords, all_results):
        if isinstance(data, Exception):
            logger.error(f"中文指南搜索异常 (keywords={kw}): {data}")
            continue
        logger.info(f"查询结果条数: {len(data)}")
        # 合并结果，按id去重，保留最高权重
        for item in data:
            item_id = item.get("id")
            if not item_id:
                continue
            if item_id in merged_results:
                if item.get("weight", 0) > merged_results[item_id].get("weight", 0):
                    merged_results[item_id] = item
            else:
                merged_results[item_id] = item
    logger.info(f"合并搜索结果，现在的数据条数: {len(merged_results)}")

    # 转换为列表并按权重排序
    data = list(merged_results.values())
    data.sort(key=lambda x: x.get("weight", 0) * x.get("reranker_score", 0), reverse=True)

    # 构建返回给模型的文本内容
    contents = "中文指南检索结果：\n"
    result_data = data[:2]

    for item in result_data:
        title = item.get("title")
        p_time = item.get("publish_time")
        weight = item.get("weight")
        logger.info(f"中文指南排序过滤结果: | 标题: {title} | 发布时间: {p_time} | ID: {item.get('id')} | 权重: {weight}")
        contents += f"标题：{title}\n发表时间：{p_time}\n权重：{weight}\n"
        for one_sentence in item.get("match_sentences", []):
            contents += f"引用标识：{one_sentence.get('id')}\n{one_sentence.get('sentence')}\n"
        contents += "\n"
    return {
        "content": contents,
        "data": {"db": "search_guidelinezh_db", "result": result_data}
    }

async def search_guideline_en(P, I, C, O) -> Dict[str, Any]:
    """
    Asynchronous search the English guideline database.
    分4次搜索：P → PI → PIC → PICO，合并结果并按id去重（保留最高权重）
    """
    search_keywords = [
        P.strip(),
        f"{I}".strip(),
        f"{P} {I}".strip(),
        f"{P} {I} {C}".strip(),
        f"{P} {I} {O}".strip(),
        f"{P} {I} {C} {O}".strip(),
    ]

    merged_results = {}

    # 构建异步任务列表，并发执行所有关键词组合搜索
    tasks = []
    valid_keywords = []
    for idx, keywords in enumerate(search_keywords):
        if not keywords.strip():
            continue
        valid_keywords.append(keywords)
        tasks.append(
            milvus_item_search(keywords=keywords,
                              collection_name="guideline_en",
                              dense_name="doc_content_chunk_dense",
                              sparse_name="doc_content_chunk_sparse",
                              content_chunk_field="doc_content_chunk",
                              output_fields=["doc_abstract","doc_title", "doc_content_chunk", "doc_publish_time", "tier", "doc_id","doc_if","framer","zky_area"],
                              num=20,
                              Threshold=0.1,
                              db_key="search_guideline_db")
        )

    # 并发执行所有搜索任务
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for kw, data in zip(valid_keywords, all_results):
        if isinstance(data, Exception):
            logger.error(f"英文指南搜索异常 (keywords={kw}): {data}")
            continue
        # 合并结果，按id去重，保留最高权重
        for item in data:
            item_id = item.get("id")
            if not item_id:
                continue
            if item_id in merged_results:
                if item.get("weight", 0) > merged_results[item_id].get("weight", 0):
                    merged_results[item_id] = item
            else:
                merged_results[item_id] = item
    
    # 转换为列表并按权重排序
    data = list(merged_results.values())
    data.sort(key=lambda x: x.get("weight", 0) * x.get("reranker_score", 0), reverse=True)
    # 过滤 weight > 1.0，如果全都在 1.0 以下则不过滤
    high_quality_data = [x for x in data if x.get("weight", 0) > 1.0]
    if high_quality_data:
        data = high_quality_data
    # 构建返回给模型的文本内容
    contents = "英文指南检索结果：\n"
    result_data = data[:2]

    for item in result_data:
        title = item.get("title")
        p_time = item.get("publish_time")
        weight = item.get("weight")
        logger.info(f"英文指南排序过滤结果: | 标题: {title} | 发布时间: {p_time} | ID: {item.get('id')} | 权重: {weight}")
        contents += f"标题：{title}\n发表时间：{p_time}\n权重：{weight}\n"
        for one_sentence in item.get("match_sentences", []):
            contents += f"引用标识：{one_sentence.get('id')}\n{one_sentence.get('sentence')}\n"
        contents += "\n"

    return {
        "content": contents,
        "data": {"db": "search_guideline_db", "result": result_data}
    }

async def search_systematic_meta_db(P, I, C, O) -> Dict[str, Any]:
    """
    search Systematic Review and Meta-Analysis database
    分4次搜索：P → PI → PIC → PICO，合并结果并按id去重（保留最高权重）
    """
    search_keywords = [
        P.strip(),
        f"{I}".strip(),
        f"{P} {I}".strip(),
        f"{P} {I} {C}".strip(),
        f"{P} {I} {O}".strip(),
        f"{P} {I} {C} {O}".strip(),
    ]

    merged_results = {}

    # 构建异步任务列表，并发执行所有关键词组合搜索
    tasks = []
    valid_keywords = []
    for idx, keywords in enumerate(search_keywords):
        if not keywords.strip():
            continue
        valid_keywords.append(keywords)
        tasks.append(
            milvus_item_search(keywords=keywords, 
                              collection_name="systematic_and_meta", 
                              dense_name="doc_content_chunk_dense", 
                              sparse_name="doc_content_chunk_sparse", 
                              content_chunk_field="doc_content_chunk", 
                              output_fields=["doc_abstract","doc_title", "doc_content_chunk", "doc_publish_time", "tier", "doc_id","doc_if","full_name_new","zky_area"],
                              num=20,
                              Threshold=0.1,
                              db_key="search_meta_db")
        )

    # 并发执行所有搜索任务
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for kw, data in zip(valid_keywords, all_results):
        if isinstance(data, Exception):
            logger.error(f"系统评价和Meta搜索异常 (keywords={kw}): {data}")
            continue
        
        # 二次过滤：标题必须包含 meta-analysis 或 systematic review
        filtered_records = []
        for record in data:
            title = record.get("title", "")
            if title and ("meta-analysis" in title.lower() or "systematic review" in title.lower()):
                filtered_records.append(record)
        
        # 合并结果，按id去重，保留最高权重
        for item in filtered_records:
            item_id = item.get("id")
            if not item_id:
                continue
            if item_id in merged_results:
                if item.get("weight", 0) > merged_results[item_id].get("weight", 0):
                    merged_results[item_id] = item
            else:
                merged_results[item_id] = item

    # 转换为列表并按权重排序
    filtered_records = list(merged_results.values())
    filtered_records.sort(key=lambda x: x.get("weight", 0) * x.get("reranker_score", 0), reverse=True)
    logger.info(f"系统评价和Meta第一次过滤结果:\n{filtered_records}")
    # 过滤 weight > 1.0，如果全都在 1.0 以下则不过滤
    high_quality_data = [x for x in filtered_records if x.get("weight", 0) > 1.0]
    if high_quality_data:
        filtered_records = high_quality_data
    # 构建返回给模型的文本内容
    contents = "系统评价和Meta分析检索结果：\n"
    result_data = filtered_records[:2]

    for item in result_data:
        title = item.get("title")
        p_time = item.get("publish_time")
        weight = item.get("weight")
        logger.info(f"系统评价和Meta分析排序过滤结果: | 标题: {title} | 发布时间: {p_time} | ID: {item.get('id')} | 权重: {weight}")
        contents += f"标题：{title}\n发表时间：{p_time}\n权重：{weight}\n"
        for one_sentence in item.get("match_sentences", []):
            contents += f"引用标识：{one_sentence.get('id')}\n{one_sentence.get('sentence')}\n"
        contents += "\n"

    return {
        "content": contents,
        "data": {"db": "search_meta_db", "result": result_data}
    }

async def search_clinical_db(P, I, C, O) -> Dict[str, Any]:
    """
    search RCT database
    分4次搜索：P → PI → PIC → PICO，合并结果并按id去重（保留最高权重）
    """
    search_keywords = [
        P.strip(),
        f"{I}".strip(),
        f"{P} {I}".strip(),
        f"{P} {I} {C}".strip(),
        f"{P} {I} {O}".strip(),
        f"{P} {I} {C} {O}".strip(),
    ]

    merged_results = {}

    # 构建异步任务列表，并发执行所有关键词组合搜索
    tasks = []
    valid_keywords = []
    for idx, keywords in enumerate(search_keywords):
        if not keywords.strip():
            continue
        valid_keywords.append(keywords)
        tasks.append(
            milvus_item_search(keywords=keywords,
                              collection_name="RCT",
                              dense_name="doc_content_chunk_dense",
                              sparse_name="doc_content_chunk_sparse",
                              content_chunk_field="doc_content_chunk",
                              output_fields=["doc_abstract","doc_title", "doc_content_chunk", "doc_publish_time", "tier", "doc_id","doc_if","full_name_new","zky_area"],
                              num=20,
                              Threshold=0.1,
                              db_key="search_clinical_db")
        )

    # 并发执行所有搜索任务
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for kw, data in zip(valid_keywords, all_results):
        if isinstance(data, Exception):
            logger.error(f"RCT搜索异常 (keywords={kw}): {data}")
            continue
        # 合并结果，按id去重，保留最高权重
        for item in data:
            item_id = item.get("id")
            if not item_id:
                continue
            if item_id in merged_results:
                if item.get("weight", 0) > merged_results[item_id].get("weight", 0):
                    merged_results[item_id] = item
            else:
                merged_results[item_id] = item

    # 转换为列表并按权重排序
    data = list(merged_results.values())
    data.sort(key=lambda x: x.get("weight", 0) * x.get("reranker_score", 0), reverse=True)

    # 过滤 weight > 1.0，如果全都在 1.0 以下则不过滤
    high_quality_data = [x for x in data if x.get("weight", 0) > 1.0]
    if high_quality_data:
        data = high_quality_data

    # 构建返回给模型的文本内容
    contents = "RCT检索结果：\n"
    result_data = data[:2]

    for item in result_data:
        title = item.get("title")
        p_time = item.get("publish_time")
        weight = item.get("weight")
        logger.info(f"RCT排序过滤结果: | 标题: {title} | 发布时间: {p_time} | ID: {item.get('id')} | 权重: {weight}")
        contents += f"标题：{title}\n发表时间：{p_time}\n权重：{weight}\n"
        for one_sentence in item.get("match_sentences", []):
            contents += f"引用标识：{one_sentence.get('id')}\n{one_sentence.get('sentence')}\n"
        contents += "\n"

    return {
        "content": contents,
        "data": {"db": "search_clinical_db", "result": result_data}
    }

@function_tool
async def search_embedding_db(P: str, I: str, C: str="", O: str=""):
    """
    从用户问题中提取 PICO，以此搜索全部数据库，禁止添加年份时间。

    :param P: P（人群）：疾病、严重程度、年龄/性别、伴随疾病、基线风险、场景（门诊/住院/ICU等）。
    :param I: I（干预）：剂量、疗程、途径、频次、联合用药/策略细节。
    :param C: 尽量提供，可以为空
    :param O: 尽量提供，可以为空
    :return: Combined JSON object with all search results
    """
    
    # Phase 1:中英文指南
    async def run_guideline():
        try:
            res = await search_guideline_zh(P,I,C,O)
            return res
        except Exception as e:
            logger.error(f"Error in search_guideline_zh: {e}")
            return None

    async def run_guideline_en():
        try:
            res = await search_guideline_en(P,I,C,O)
            return res
        except Exception as e:
            logger.error(f"Error in run_guideline_en: {e}")
            return None

    # Concurrent Search for all databases
    async def run_systematic_meta():
        try:
            res = await search_systematic_meta_db(P,I,C,O)
            return res
        except Exception as e:
            logger.error(f"Error in search_guidelinezh_db: {e}")
            return None

    async def run_clinical():
        try:
            res = await search_clinical_db(P,I,C,O)
            return res
        except Exception as e:
            logger.error(f"Error in search_clinical_db: {e}")
            return None

    results = await asyncio.gather(run_guideline(), run_guideline_en(), run_systematic_meta(), run_clinical())
    
    # 分别收集返回给模型的文本内容和存储到state的数据
    all_contents = ""
    
    for res in results:
        if res:
            all_contents += res.get("content", "")

    logger.info("search_all_dbs completed. ====================")
    # logger.info(f"返回给模型的文本内容:\n{all_contents}")
    # 返回给模型的是易读的文本格式
    return all_contents


if __name__ == "__main__":
    async def my_test_meta():
        # 创建一个假的工具上下文（只需包含 state）
        print("开始测试 search_embedding_db 并验证 SoF 数据注入...")
        mock_context = MockToolContext()
        
        result = await search_embedding_db(
            P="心脏骤停", I="心肺复苏", C="", O="",
            tool_context=mock_context
        )
        print(f"搜索结果内容：{result}")
    # 运行异步测试函数
    asyncio.run(my_test_meta())