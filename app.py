import os
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="AI 求职 Agent · 石丁", page_icon="🎯", layout="wide")

load_dotenv()  # 本地开发：先读 .env，后面 secrets 可覆盖


def _env(name: str, default: str = "") -> str:
    """Streamlit Secrets 优先，其次环境变量。"""
    try:
        v = st.secrets[name]
        if v:
            return str(v)
    except Exception:
        pass
    return os.getenv(name, default)


# 只在有值时才覆盖，否则会把 .env 里的值冲成空串
for _k, _d in [
    ("DEEPSEEK_API_KEY", ""),
    ("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    ("LLM_MODEL", "deepseek-chat"),
]:
    _v = _env(_k, _d)
    if _v:
        os.environ[_k] = _v

from job_agent import build_agent, JOBS, SKILL_DB, DATA_DATE  # noqa: E402

st.title("🎯 AI 求职 Agent")
st.caption(
    "LangChain 1.x `create_agent` + ReAct · 4 个工具 · 504 条岗位 JD　|　"
    "github.com/Eria09/job_agent"
)

with st.sidebar:
    st.markdown("### 📊 数据")
    st.metric("岗位 JD", f"{len(JOBS):,} 条")
    st.metric("技能维度", f"{len(SKILL_DB)} 个")
    st.caption(f"数据截至 {DATA_DATE}，自采实习僧")
    st.divider()
    st.markdown("### 🛠 4 个工具")
    st.markdown(
        "- `query_skill` 技能需求度\n"
        "- `query_salary` 薪资区间\n"
        "- `search_jobs` 岗位检索（含真实链接）\n"
        "- `compare_skills` 技能对比"
    )
    st.divider()
    st.caption("首次加载约需 10–20 秒（模型与数据初始化）")

if "n" not in st.session_state:
    st.session_state.n = 0
MAX_TURNS = 20

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    if isinstance(m, dict):
        role, content = m.get("role", "user"), m.get("content", "")
    else:
        role = "user" if m.type == "human" else "assistant"
        content = m.content
    with st.chat_message(role):
        st.markdown(content)

st.caption(
    "试试：`RAG 和 Agent 哪个更值得先学？`　|　`北京有哪些本科能投的 Agent 岗`　|　"
    "`AI 应用开发实习多少钱一天`"
)

q = st.chat_input("问点什么…")
if q:
    if st.session_state.n >= MAX_TURNS:
        st.warning("本会话额度已用完，刷新页面重新开始 🙂")
        st.stop()
    st.session_state.n += 1

    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("调用工具中…"):
            agent = build_agent()
            res = agent.invoke(
                {"messages": st.session_state.msgs},
                config={"recursion_limit": 12},
            )
            ans = res["messages"][-1].content or "(模型没有返回文字内容，可能停在工具调用)"
        st.markdown(ans)

    st.session_state.msgs = res["messages"]
