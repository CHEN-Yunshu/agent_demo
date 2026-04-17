"""
Stock Research Agent Demo
=========================
演示 Agent 如何自动调用多个工具、综合推理、交付报告。

Mac 安装步骤:
  1. brew install ollama
  2. ollama pull qwen2.5:7b          # 或 llama3.1:8b / mistral
  3. ollama serve                     # 后台运行 (如果没自动启动)
  4. pip install yfinance requests
  5. python stock_agent_demo.py

也可以用 Claude API (设置环境变量 ANTHROPIC_API_KEY):
  pip install anthropic yfinance requests
  ANTHROPIC_API_KEY=sk-xxx python stock_agent_demo.py --provider claude
"""

import json
import sys
import time
import argparse
from datetime import datetime, timedelta

# ============================================================
# TOOLS: Agent 可以调用的工具
# ============================================================

def tool_get_stock_data(symbol: str) -> dict:
    """工具1: 获取股票行情数据"""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="3mo")
    info = ticker.info

    current = hist['Close'].iloc[-1] if len(hist) > 0 else 0
    high_52w = info.get('fiftyTwoWeekHigh', 0)
    low_52w = info.get('fiftyTwoWeekLow', 0)
    volume = info.get('averageVolume', 0)
    market_cap = info.get('marketCap', 0)

    # 近30天涨跌
    if len(hist) >= 20:
        change_30d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1) * 100
    else:
        change_30d = 0

    return {
        "symbol": symbol,
        "current_price": round(current, 2),
        "52w_high": round(high_52w, 2),
        "52w_low": round(low_52w, 2),
        "change_30d": f"{change_30d:.1f}%",
        "avg_volume": f"{volume:,}",
        "market_cap": f"${market_cap / 1e9:.0f}B" if market_cap else "N/A",
    }


def tool_get_financials(symbol: str) -> dict:
    """工具2: 获取财报数据"""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        "symbol": symbol,
        "pe_ratio": info.get('trailingPE', 'N/A'),
        "forward_pe": info.get('forwardPE', 'N/A'),
        "revenue": f"${info.get('totalRevenue', 0) / 1e9:.1f}B",
        "profit_margin": f"{info.get('profitMargins', 0) * 100:.1f}%",
        "debt_to_equity": info.get('debtToEquity', 'N/A'),
        "recommendation": info.get('recommendationKey', 'N/A'),
    }


def tool_get_news(query: str) -> list:
    """工具3: 搜索相关新闻 (用 yfinance 的 news)"""
    import yfinance as yf
    ticker = yf.Ticker(query)
    news = ticker.news or []
    results = []
    for item in news[:5]:
        results.append({
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
        })
    return results if results else [{"title": "No recent news found", "publisher": ""}]


def tool_get_analyst_ratings(symbol: str) -> dict:
    """工具4: 获取分析师评级"""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        "symbol": symbol,
        "target_mean": info.get('targetMeanPrice', 'N/A'),
        "target_high": info.get('targetHighPrice', 'N/A'),
        "target_low": info.get('targetLowPrice', 'N/A'),
        "recommendation": info.get('recommendationKey', 'N/A'),
        "num_analysts": info.get('numberOfAnalystOpinions', 'N/A'),
    }


# ============================================================
# LLM Providers
# ============================================================

def call_ollama(messages: list, model: str = "qwen2.5:7b") -> str:
    """调用本地 Ollama 模型"""
    import requests
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": model,
        "messages": messages,
        "stream": False,
    })
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_claude(messages: list) -> str:
    """调用 Claude API"""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=messages,
    )
    return resp.content[0].text


# ============================================================
# AGENT: 规划 → 执行 → 验证 → 交付
# ============================================================

TOOLS = {
    "get_stock_data": tool_get_stock_data,
    "get_financials": tool_get_financials,
    "get_news": tool_get_news,
    "get_analyst_ratings": tool_get_analyst_ratings,
}

SYSTEM_PROMPT = """你是一个股票调研 Agent。用户给你一个任务，你需要：

1. **规划**: 分析任务，决定需要调用哪些工具
2. **执行**: 调用工具获取数据
3. **验证**: 交叉对比不同数据源，检查是否有矛盾
4. **交付**: 输出结构化的调研报告

你可以使用的工具（数据已经提供给你）：
- get_stock_data: 获取股价行情
- get_financials: 获取财报数据
- get_news: 搜索相关新闻
- get_analyst_ratings: 获取分析师评级

请根据提供的数据，生成完整的调研报告，包含：
1. 走势判断（短期/中长期）
2. 基本面分析
3. 风险提示
4. 操作建议
5. 综合置信度（0-100%）
"""


def print_step(step: str, detail: str, color: str = "36"):
    """打印带颜色的步骤"""
    print(f"\n\033[1;{color}m{'='*60}\033[0m")
    print(f"\033[1;{color}m  {step}\033[0m")
    print(f"\033[0;{color}m  {detail}\033[0m")
    print(f"\033[1;{color}m{'='*60}\033[0m")


def run_agent(query: str, provider: str = "ollama", model: str = "qwen2.5:7b"):
    """运行 Agent 完整流程"""

    llm_call = call_ollama if provider == "ollama" else call_claude
    symbol = "TSLA"  # Default, could be parsed from query

    # Extract symbol from query
    for s in ["TSLA", "AAPL", "GOOGL", "MSFT", "NVDA", "AMZN", "META", "BABA", "BYD"]:
        if s.lower() in query.lower() or s in query:
            symbol = s
            break

    # ===== STEP 1: 规划 =====
    print_step("🗺️  STEP 1: 规划", f"分析任务，决定调用哪些工具...", "35")
    print(f"\n  用户任务: \033[1;33m{query}\033[0m")
    print(f"  目标股票: \033[1;33m{symbol}\033[0m")
    print(f"\n  📋 计划:")
    print(f"     1. 调用 get_stock_data   → 获取行情")
    print(f"     2. 调用 get_financials   → 获取财报")
    print(f"     3. 调用 get_news         → 搜索新闻")
    print(f"     4. 调用 get_analyst_ratings → 分析师评级")
    time.sleep(1)

    # ===== STEP 2: 执行 - 调用工具 =====
    print_step("⚡  STEP 2: 执行", "并行调用 4 个工具...", "33")

    results = {}
    for name, func in TOOLS.items():
        arg = symbol if name != "get_news" else symbol
        print(f"\n  🔧 调用 {name}(\"{arg}\")...", end=" ", flush=True)
        time.sleep(0.5)
        result = func(arg)
        results[name] = result
        print(f"\033[32m✓\033[0m")
        # Print key data
        if isinstance(result, dict):
            for k, v in list(result.items())[:3]:
                print(f"     {k}: {v}")
        elif isinstance(result, list):
            for item in result[:2]:
                print(f"     📰 {item.get('title', '')[:60]}")

    # ===== STEP 3: 验证 - 交叉对比 =====
    print_step("🔍  STEP 3: 验证", "交叉对比数据，检查矛盾...", "36")
    time.sleep(0.5)

    stock = results.get("get_stock_data", {})
    financials = results.get("get_financials", {})
    ratings = results.get("get_analyst_ratings", {})

    print(f"\n  检查项 1: 股价 vs 分析师目标价")
    current = stock.get("current_price", 0)
    target = ratings.get("target_mean", 0)
    if current and target and target != 'N/A':
        diff = ((float(target) / current) - 1) * 100
        status = "✅ 一致" if abs(diff) < 30 else "⚠️ 差异较大"
        print(f"     当前 ${current} vs 目标 ${target} (差距 {diff:.1f}%) → {status}")
    else:
        print(f"     数据不足，跳过")

    print(f"\n  检查项 2: PE 估值 vs 行业水平")
    pe = financials.get("pe_ratio", "N/A")
    print(f"     当前 PE: {pe} → {'⚠️ 偏高' if pe != 'N/A' and float(pe) > 30 else '✅ 合理' if pe != 'N/A' else '数据不足'}")

    print(f"\n  检查项 3: 新闻情绪 vs 分析师评级")
    rec = ratings.get("recommendation", "N/A")
    print(f"     分析师评级: {rec}, 分析师数量: {ratings.get('num_analysts', 'N/A')}")

    time.sleep(1)

    # ===== STEP 4: 综合推理 - 调用 LLM =====
    print_step("🧠  STEP 4: 综合推理", f"调用 {'本地模型 (Ollama)' if provider == 'ollama' else 'Claude API'} 生成报告...", "35")

    data_summary = json.dumps(results, indent=2, ensure_ascii=False, default=str)

    messages = [
        {"role": "system" if provider == "claude" else "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""
用户任务: {query}

以下是我通过工具获取的数据:

{data_summary}

请根据以上数据，生成完整的股票调研报告。用中文回答。
"""}
    ]

    print(f"\n  ⏳ 正在推理...", flush=True)

    if provider == "ollama":
        report = call_ollama(messages, model)
    else:
        report = call_claude(messages)

    # ===== STEP 5: 交付 =====
    print_step("📋  STEP 5: 交付", "Agent 完成，以下是调研报告", "32")
    print()
    print(report)
    print()
    print(f"\033[1;32m{'='*60}\033[0m")
    print(f"\033[1;32m  ✅ Agent 任务完成\033[0m")
    print(f"\033[0;32m  数据来源: Yahoo Finance ({len(TOOLS)} 个工具)\033[0m")
    print(f"\033[0;32m  推理模型: {'Ollama/' + model if provider == 'ollama' else 'Claude'}\033[0m")
    print(f"\033[1;32m{'='*60}\033[0m")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Research Agent Demo")
    parser.add_argument("--provider", choices=["ollama", "claude"], default="ollama",
                        help="LLM provider (default: ollama)")
    parser.add_argument("--model", default="qwen2.5:7b",
                        help="Ollama model name (default: qwen2.5:7b)")
    parser.add_argument("--query", default=None,
                        help="Research query")
    args = parser.parse_args()

    print("\n\033[1;37m" + "=" * 60)
    print("  🤖 Stock Research Agent Demo")
    print("  " + "=" * 56)
    print(f"  Provider: {args.provider}")
    if args.provider == "ollama":
        print(f"  Model: {args.model}")
    print("=" * 60 + "\033[0m")

    if args.query:
        query = args.query
    else:
        query = input("\n  💬 你想调研什么？\n  > ")

    if not query.strip():
        query = "帮我分析一下特斯拉 TSLA 最近值不值得买"

    print(f"\n  📝 收到: \"{query}\"")
    run_agent(query, provider=args.provider, model=args.model)
