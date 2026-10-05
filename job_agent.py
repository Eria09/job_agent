"""
AI 求职辅助 Agent
==================
基于 LangChain 1.x create_agent + ReAct 范式，封装 3 个工具：

  1. query_skill   查询某个技能在 AI 岗位中的需求程度
  2. query_salary  按岗位关键词统计薪资区间
  3. search_jobs   按城市 / 学历 / 技能检索具体岗位

数据来源：自采 504 条 AI 相关实习岗位 JD（实习僧）
运行：python job_agent.py
"""
import json
import os
import re
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

# ---------------- 配置 ----------------
load_dotenv()

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

BASE_DIR = Path(__file__).parent
SKILL_FILE = BASE_DIR / "data" / "skill_kb.json"
JOBS_FILE = BASE_DIR / "data" / "jobs.json"

if not SKILL_FILE.exists() or not JOBS_FILE.exists():
    raise SystemExit(
        "❌ 缺少数据文件\n"
        f"   请确认以下文件存在：\n"
        f"     {SKILL_FILE}\n"
        f"     {JOBS_FILE}"
    )

# ---------------- 加载数据 ----------------
_skill_raw = json.loads(SKILL_FILE.read_text(encoding="utf-8"))
SKILL_DB = {s["skill"].lower(): s for s in _skill_raw["skills"]}
SAMPLE_SIZE = _skill_raw["sample_size"]

JOBS = json.loads(JOBS_FILE.read_text(encoding="utf-8"))

print(f"[数据] 技能库 {len(SKILL_DB)} 个技能（样本 {SAMPLE_SIZE} 条 JD）")
print(f"[数据] 岗位库 {len(JOBS)} 条岗位")


# ---------------- 工具 1：技能需求查询 ----------------
@tool
def query_skill(skill: str) -> str:
    """查询某个技术技能在 AI 岗位中的需求程度（提及率、岗位数、典型岗位）。

    Args:
        skill: 技能名称，例如 Python、RAG、Agent、Docker、MySQL
    """
    key = skill.strip().lower()
    hit = SKILL_DB.get(key)
    if hit is None:  # 模糊匹配
        cands = [v for k, v in SKILL_DB.items() if key in k or k in key]
        if not cands:
            top = "、".join(list(SKILL_DB)[:20])
            return f"技能库中没有「{skill}」。目前可查：{top} ……"
        hit = max(cands, key=lambda x: x["count"])

    jobs = "、".join(hit["jobs"][:5]) if hit.get("jobs") else "—"
    return (
        f"【{hit['skill']}】\n"
        f"- 提及岗位数：{hit['count']} / {SAMPLE_SIZE} 条\n"
        f"- 需求度：{hit['ratio']:.1%}\n"
        f"- 典型岗位：{jobs}"
    )


# ---------------- 工具 2：薪资查询 ----------------
def _parse_salary(text: str):
    """从 '200-300/天' 中解析出 [200, 300]（单位：元/天）"""
    m = re.search(r"(\d+)\s*-\s*(\d+)", text or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)", text or "")
    if m:
        v = int(m.group(1))
        return v, v
    return None


@tool
def query_salary(job_keyword: str) -> str:
    """按岗位关键词统计薪资区间（元/天），返回中位数与区间。

    Args:
        job_keyword: 岗位关键词，例如 AI应用开发、算法、后端、数据分析
    """
    kw = job_keyword.strip().lower()
    matched = [j for j in JOBS if kw in j["title"].lower()]
    if not matched:
        return f"没有找到标题包含「{job_keyword}」的岗位，共 {len(JOBS)} 条岗位可供检索。"

    lows, highs = [], []
    for j in matched:
        p = _parse_salary(j.get("salary", ""))
        if p:
            lows.append(p[0]); highs.append(p[1])
    if not lows:
        return f"找到 {len(matched)} 个「{job_keyword}」岗位，但薪资均为「面议」。"

    return (
        f"【{job_keyword}】共 {len(matched)} 个岗位（有薪资标注 {len(lows)} 个）\n"
        f"- 日薪区间：{min(lows)} ~ {max(highs)} 元/天\n"
        f"- 中位区间：{int(statistics.median(lows))} ~ {int(statistics.median(highs))} 元/天\n"
        f"- 样本岗位：{'、'.join(j['title'][:18] for j in matched[:4])}"
    )


# ---------------- 工具 3：岗位检索 ----------------
@tool
def search_jobs(city: str = "", education: str = "", skill: str = "", limit: int = 5) -> str:
    """按城市 / 学历 / 技能关键词检索匹配的实习岗位。

    Args:
        city: 城市名，例如 北京、上海、长沙；留空表示不限
        education: 学历要求，例如 本科、硕士；留空表示不限
        skill: 技能关键词，例如 RAG、Agent、Python、Docker
        limit: 最多返回几条，默认 5
    """
    res = JOBS
    if city.strip():
        res = [j for j in res if city.strip() in j["city"]]
    if education.strip():
        if education.strip() == "本科":
            res = [j for j in res if j["education"] != "硕士"]  # 本科可投
        else:
            res = [j for j in res if education.strip() in j["education"]]
    if skill.strip():
        s = skill.strip().lower()
        res = [j for j in res if any(s in x.lower() for x in j["skills"])]

    if not res:
        return "没有匹配的岗位，建议放宽条件（例如去掉城市或学历限制）。"

    lines = [f"共匹配 {len(res)} 个岗位，列出前 {min(limit, len(res))} 个："]
    for j in res[:limit]:
        skills = "、".join(j["skills"][:6]) or "—"
        lines.append(
            f"\n▪ {j['title']}\n"
            f"   城市：{j['city']}　薪资：{j['salary']}　学历：{j['education']}\n"
            f"   技能：{skills}\n"
            f"   链接：{j['link'] or '—'}"
        )
    return "\n".join(lines)


# ---------------- Agent ----------------
SYSTEM_PROMPT = """你是一位资深的 AI 求职顾问，专门帮助求职者分析 AI / 大模型 / Agent 方向的实习岗位市场。

你可以调用三个工具：
1. query_skill(skill)                        —— 查询某项技能在岗位中的需求程度
2. query_salary(job_keyword)                 —— 查询某类岗位的薪资区间
3. search_jobs(city, education, skill, limit) —— 按条件检索具体岗位

工作原则：
- 必须先调用工具获取真实数据，再基于数据回答，禁止凭记忆编造数字
- 回答时引用具体数据（提及率、岗位数、薪资区间）
- 如果查询的技能不存在，如实说明，并推荐相近技能
- 用中文回答，简洁、分点，不要长篇大论
"""


def build_agent():
    if not DEEPSEEK_KEY:
        raise SystemExit("❌ 没读到 DEEPSEEK_API_KEY，请检查 .env 文件")
    llm = ChatDeepSeek(
        model=MODEL,
        api_key=DEEPSEEK_KEY,
        api_base=DEEPSEEK_URL,
        temperature=0,
    )
    return create_agent(
        llm,
        tools=[query_skill, query_salary, search_jobs],
        system_prompt=SYSTEM_PROMPT,
    )


# ---------------- 自检模式（不调用大模型） ----------------
def self_test():
    print("\n" + "=" * 62)
    print("工具自检（不需要大模型）")
    print("=" * 62)
    print("\n[1] query_skill('RAG')")
    print(query_skill.invoke({"skill": "RAG"}))
    print("\n[2] query_salary('AI应用开发')")
    print(query_salary.invoke({"job_keyword": "AI应用开发"}))
    print("\n[3] search_jobs(city='北京', education='本科', skill='Agent', limit=3)")
    print(search_jobs.invoke({"city": "北京", "education": "本科", "skill": "Agent", "limit": 3}))
    print("\n✅ 工具自检完成")


# ---------------- 主流程 ----------------
def main():
    if "--test" in sys.argv:
        self_test()
        return

    agent = build_agent()
    print("=" * 62)
    print("AI 求职辅助 Agent  ·  输入 q 退出")
    print("试试问：RAG 这个技能市场需求怎么样？")
    print("=" * 62)

    while True:
        q = input("\n你问：").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": q}]})
            print("\n回答：", result["messages"][-1].content)
        except Exception as e:
            print("❌ 出错：", type(e).__name__, e)


if __name__ == "__main__":
    main()
