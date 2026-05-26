#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedAgent 训练日志分析工具
用途：通过 SSH 连接服务器，分析 v17/v18 训练日志
注意：运行前确保 paramiko 已安装（pip install paramiko）
敏感信息：服务器凭据请从环境变量读取，不要硬编码
"""

import os
import re
import sys
import json
import io
import paramiko
from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================
# 服务器配置（从环境变量读取，避免硬编码）
# ============================================================

HOST = os.getenv("MEDAGENT_SERVER_HOST", "")
PORT = int(os.getenv("MEDAGENT_SERVER_PORT", "22"))
USER = os.getenv("MEDAGENT_SERVER_USER", "root")
PASS = os.getenv("MEDAGENT_SERVER_PASS", "")  # 建议使用 SSH key 替代密码

# ============================================================
# 日志文件路径
# ============================================================

LOG_V17 = "/tmp/train_v17_plus_small.log"
LOG_V18 = "/tmp/train_v18_full_grpo.log"
PROGRESS_V17 = "/tmp/sql_agent_progress_v17_plus_small.json"
PROGRESS_V18 = "/tmp/sql_agent_progress_v18_full_grpo.json"


def ssh_run(cmd: str, timeout: int = 120) -> Tuple[str, str]:
    """执行远程 SSH 命令，返回 (stdout, stderr)"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, PORT, USER, PASS, timeout=30)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    return out, err


def parse_total_rewards(log_lines: str) -> List[float]:
    """从日志中提取所有 Total Reward 值"""
    pattern = re.compile(r"\[Total Reward\] ([\d.]+)")
    rewards = []
    for m in pattern.finditer(log_lines):
        try:
            rewards.append(float(m.group(1)))
        except ValueError:
            pass
    return rewards


def parse_evidence_safety(log_lines: str) -> Dict[str, int]:
    """统计 EvidenceSafety 分数分布"""
    pattern = re.compile(r"\[EvidenceSafety\] strong=(\w+), cautious=(\w+), prefixes=\{([^}]*)\}, score=([-.\d]+)")
    stats = defaultdict(int)
    for m in pattern.finditer(log_lines):
        score = float(m.group(4))
        if score < 0:
            stats["negative"] += 1
        elif score > 0:
            stats["positive"] += 1
        else:
            stats["zero"] += 1
    return dict(stats)


def analyze_log(version: str, log_path: str) -> None:
    """分析指定版本的训练日志"""
    print(f"\n{'='*60}")
    print(f"分析 {version} 日志: {log_path}")
    print(f"{'='*60}")

    if not HOST or not PASS:
        print("[ERROR] 请设置环境变量 MEDAGENT_SERVER_HOST 和 MEDAGENT_SERVER_PASS")
        return

    # 获取最后1000行
    out, err = ssh_run(f"tail -1000 {log_path} 2>/dev/null || echo FILE_NOT_FOUND")
    if "FILE_NOT_FOUND" in out:
        print(f"[WARN] 日志文件未找到: {log_path}")
        return

    rewards = parse_total_rewards(out)
    if rewards:
        print(f"Total Reward 统计（最后1000行）:")
        print(f"  数量: {len(rewards)}")
        print(f"  均值: {sum(rewards)/len(rewards):.3f}")
        print(f"  最大: {max(rewards):.3f}")
        print(f"  最小: {min(rewards):.3f}")
    else:
        print("未找到 Total Reward 记录")

    if version == "v18":
        es_stats = parse_evidence_safety(out)
        if es_stats:
            print(f"\nEvidenceSafety 分布（最后1000行）:")
            total = sum(es_stats.values())
            for k, v in es_stats.items():
                pct = v / total * 100 if total > 0 else 0
                print(f"  {k}: {v} ({pct:.1f}%)")

    # 统计 no_tool 事件
    no_tool_out, _ = ssh_run(f"grep 'no_tool' {log_path} | wc -l")
    print(f"\nno_tool 事件数: {no_tool_out.strip()}")

    # 统计 error
    error_out, _ = ssh_run(f"grep -i 'error\\|fatal\\|exception' {log_path} | wc -l")
    print(f"error/fatal/exception 行数: {error_out.strip()}")


def get_progress(version: str, progress_path: str) -> None:
    """读取进度文件"""
    print(f"\n{'='*60}")
    print(f"{version} 进度状态: {progress_path}")
    print(f"{'='*60}")

    if not HOST or not PASS:
        print("[ERROR] 请设置环境变量")
        return

    out, err = ssh_run(f"cat {progress_path} 2>/dev/null || echo FILE_NOT_FOUND")
    if "FILE_NOT_FOUND" in out:
        print(f"[WARN] 进度文件未找到")
        return

    try:
        data = json.loads(out.strip())
        for k, v in data.items():
            print(f"  {k}: {v}")
    except json.JSONDecodeError:
        print(out)


def main():
    """主入口"""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if not HOST:
        print("请设置 MEDAGENT_SERVER_HOST 环境变量")
        print("示例: export MEDAGENT_SERVER_HOST=117.50.x.x")
        sys.exit(1)

    print("MedAgent 训练日志分析工具")
    print(f"目标服务器: {HOST}:{PORT}")

    get_progress("v17", PROGRESS_V17)
    get_progress("v18", PROGRESS_V18)

    analyze_log("v17", LOG_V17)
    analyze_log("v18", LOG_V18)

    print("\n分析完成。")


if __name__ == "__main__":
    main()
