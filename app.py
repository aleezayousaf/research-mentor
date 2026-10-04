import os

import streamlit as st
from crewai import Crew, Process, Task

from agents.citation_integrator import get_citation_integrator
from agents.journal_publisher import get_journal_publisher
from agents.legal_researcher import get_legal_researcher
from agents.socratic_supervisor import get_socratic_supervisor
from agents.writing_coach import get_writing_coach
from config import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    get_llm,
    setup_environment,
    test_api_key,
)


_MODEL_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def normalize_model_input(text: str) -> str:
    """Normalize typographic punctuation that breaks ASCII-only agent integrations."""
    return text.translate(_MODEL_PUNCTUATION)


st.set_page_config(
    page_title="ResearchMentor AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKFLOWS = {
    "Refine a research question": {
        "agent": get_socratic_supervisor,
        "icon": ":material/lightbulb:",
        "specialist": "Question design",
        "description": "Pressure-test your topic and find a focused, researchable question.",
        "output": "Three Socratic questions and guidance to narrow your scope.",
        "placeholder": "For example: How effectively does Pakistan's data protection framework protect biometric data?",
        "task": lambda text: (
            f"Evaluate this research idea: {text}\n"
            "GUARDRAIL ENFORCEMENT: Do NOT write paper text for the student.\n"
            "1. Ask 3 sharp Socratic questions challenging the scope and legal claims.\n"
            "2. Require the student to narrow their normative research question."
        ),
        "expected": "Socratic critique and scope-narrowing questions.",
    },
    "Discover legal sources": {
        "agent": get_legal_researcher,
        "icon": ":material/travel_explore:",
        "specialist": "Source discovery",
        "description": "Find scholarly literature and identify where primary legal sources need checking.",
        "output": "OpenAlex search results plus a clear note on official-source verification.",
        "placeholder": "For example: Pakistani case law and scholarship on freedom of expression online",
        "task": lambda text: (
            f"Search for sources related to: {text}\n"
            "1. Call 'Search Pakistan-Affiliated Scholarship (OpenAlex)' first for Pakistan-based scholarship.\n"
            "2. Call 'Search Global Scholarly Literature (OpenAlex RAG)' for worldwide scholarship.\n"
            "3. Call 'Pakistan Code Official-Source Verification Guide' for Pakistani primary law.\n"
            "4. Distinguish retrieved records from claims that still need verification. "
            "Return a source matrix and never invent statutes, cases, or citations."
        ),
        "expected": "A source matrix with retrieved records and verification notes.",
    },
    "Review an IRAC draft": {
        "agent": get_writing_coach,
        "icon": ":material/rule:",
        "specialist": "Legal writing",
        "description": "Check how your legal analysis moves from issue and rule to application and conclusion.",
        "output": "IRAC structure feedback, logical gaps, and unsupported legal leaps.",
        "placeholder": "Paste a paragraph or a short section of your draft...",
        "task": lambda text: (
            f"Analyze this draft using IRAC: {text}\n"
            "Tag paragraphs as [Issue], [Rule], [Application], or [Conclusion]. "
            "Flag logical gaps and unsupported legal leaps."
        ),
        "expected": "Detailed IRAC structural critique.",
    },
    "Format legal citations": {
        "agent": get_citation_integrator,
        "icon": ":material/format_quote:",
        "specialist": "Citation integrity",
        "description": "Turn rough legal references into consistent OSCOLA and Bluebook-style citations.",
        "output": "Formatted citation suggestions with any missing details called out.",
        "placeholder": "Paste raw citations or references, including any known court, date, or report details...",
        "task": lambda text: (
            f"Convert these raw references into OSCOLA and Bluebook footnote formats: {text}. "
            "Do not invent missing citation details; identify anything that needs checking. "
            "For journal articles or books, call 'Verify Reference with Crossref' and report whether a match was found."
        ),
        "expected": "Formatted footnote citations and a list of details needing verification.",
    },
    "Prepare for journal submission": {
        "agent": get_journal_publisher,
        "icon": ":material/publish:",
        "specialist": "Journal readiness",
        "description": "Assess a manuscript's journal fit and prepare it for double-blind review.",
        "output": "Journal-fit criteria and a practical anonymization checklist.",
        "placeholder": "Describe your article topic, method, jurisdiction, or target journals...",
        "task": lambda text: (
            f"Evaluate this topic or manuscript summary against law journal standards: {text}\n"
            "Provide a double-blind anonymization checklist and journal-fit criteria. "
            "Call 'Find Journals Publishing Similar Work (OpenAlex)' twice: region='pakistan' and region='world'. "
            "Do not claim a journal's current indexing status without a verifiable source."
        ),
        "expected": "Journal selection guidance and anonymization checklist.",
    },
}

st.html("""
<style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 12%, rgba(255, 205, 137, 0.34), transparent 24%),
            radial-gradient(ellipse at 92% 27%, rgba(134, 207, 205, 0.33), transparent 28%),
            linear-gradient(145deg, #fbf8f0 0%, #f1f6f3 48%, #f2f4fb 100%);
    }
    header[data-testid="stHeader"] { background: rgba(251, 248, 240, 0.78); }
    .stMainBlockContainer { max-width: 1240px; padding-top: 1.5rem; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(35, 75, 78, 0.12); }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.85rem; }

    #research-mentor-hero {
        position: relative;
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(210px, 0.8fr);
        align-items: center;
        min-height: 300px;
        margin: 4px 0 25px;
        padding: clamp(28px, 5vw, 52px);
        overflow: hidden;
        color: white;
        border: 1px solid rgba(255,255,255,0.52);
        border-radius: 30px;
        background:
            radial-gradient(ellipse at 80% 14%, rgba(255, 189, 97, 0.67), transparent 25%),
            radial-gradient(ellipse at 76% 105%, rgba(53, 191, 175, 0.74), transparent 34%),
            linear-gradient(122deg, #133c46 0%, #17636a 48%, #5661a4 100%);
        box-shadow: 0 28px 52px rgba(25, 61, 68, 0.24), 0 8px 18px rgba(25, 61, 68, 0.12);
        isolation: isolate;
    }
    #research-mentor-hero::before {
        position: absolute;
        z-index: -1;
        top: -125px;
        right: 13%;
        width: 330px;
        height: 330px;
        content: "";
        border: 1px solid rgba(255,255,255,0.17);
        border-radius: 50%;
        box-shadow: 0 0 0 20px rgba(255,255,255,0.035), 0 0 0 44px rgba(255,255,255,0.025);
    }
    #research-mentor-hero::after {
        position: absolute;
        z-index: -1;
        right: 2%;
        bottom: -100px;
        width: 250px;
        height: 250px;
        content: "";
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 50%;
        box-shadow: 0 0 0 15px rgba(255,255,255,0.035), 0 0 0 32px rgba(255,255,255,0.025);
    }
    #research-mentor-hero .rm-copy { position: relative; z-index: 2; max-width: 630px; }
    #research-mentor-hero .rm-kicker {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 19px;
        padding: 8px 13px;
        color: #e9fff8;
        border: 1px solid rgba(234,255,247,0.3);
        border-radius: 999px;
        background: rgba(235,255,247,0.13);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.11em;
    }
    #research-mentor-hero .rm-kicker-mark {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #91f0c8;
        box-shadow: 0 0 13px rgba(145,240,200,0.9);
    }
    #research-mentor-hero h1 {
        max-width: 620px;
        margin: 0 0 15px;
        color: white;
        font-size: clamp(2.25rem, 4.4vw, 3.65rem);
        font-weight: 780;
        letter-spacing: -0.055em;
        line-height: 1.02;
    }
    #research-mentor-hero h1 span { color: #ffd185; }
    #research-mentor-hero p {
        max-width: 535px;
        margin: 0;
        color: #e2f2f1;
        font-size: 1.03rem;
        line-height: 1.65;
    }
    #research-mentor-hero .rm-art {
        position: relative;
        display: grid;
        min-height: 225px;
        place-items: center;
        perspective: 950px;
    }
    #research-mentor-hero .rm-rays {
        position: absolute;
        width: 210px;
        height: 210px;
        border: 1px solid rgba(239,255,255,0.38);
        border-radius: 50%;
        box-shadow: 0 0 0 16px rgba(239,255,255,0.055), 0 0 0 35px rgba(239,255,255,0.035);
        transform: rotateX(68deg) rotateZ(-27deg);
    }
    #research-mentor-hero .rm-book-stack {
        position: relative;
        width: 186px;
        height: 185px;
        transform-style: preserve-3d;
        transform: rotateY(-21deg) rotateX(12deg);
        animation: rm-book-drift 5.5s ease-in-out infinite;
    }
    #research-mentor-hero .rm-book {
        position: absolute;
        width: 142px;
        height: 104px;
        border: 1px solid rgba(255,255,255,0.65);
        border-radius: 8px 13px 13px 8px;
        box-shadow: 10px 15px 18px rgba(12,34,45,0.23), inset 9px 0 0 rgba(255,255,255,0.25);
        transform: skewY(-8deg) rotateZ(-9deg);
    }
    #research-mentor-hero .rm-book::before {
        position: absolute;
        top: 7px;
        bottom: 7px;
        left: 13px;
        width: 3px;
        content: "";
        border-radius: 5px;
        background: rgba(255,255,255,0.55);
    }
    #research-mentor-hero .rm-book.book-back {
        top: 74px;
        left: 27px;
        background: linear-gradient(145deg, #f9b96d, #ed7956);
        transform: translateZ(-26px) skewY(-8deg) rotateZ(8deg);
    }
    #research-mentor-hero .rm-book.book-mid {
        top: 46px;
        left: 34px;
        background: linear-gradient(145deg, #957ee3, #6258b6);
        transform: translateZ(-10px) skewY(-8deg) rotateZ(-4deg);
    }
    #research-mentor-hero .rm-book.book-front {
        top: 20px;
        left: 23px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 17px 17px 14px 28px;
        background: linear-gradient(145deg, #fffdf5 0%, #f4ead1 100%);
        transform: translateZ(12px) skewY(-8deg) rotateZ(-12deg);
    }
    #research-mentor-hero .rm-book-title {
        color: #18434d;
        font-family: Georgia, serif;
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1;
    }
    #research-mentor-hero .rm-book-rule {
        display: block;
        width: 80%;
        height: 4px;
        border-radius: 8px;
        background: #c8d9d0;
    }
    #research-mentor-hero .rm-book-rule.short { width: 55%; }
    #research-mentor-hero .rm-float-chip {
        position: absolute;
        right: -10px;
        bottom: 22px;
        display: grid;
        width: 52px;
        height: 52px;
        place-items: center;
        color: #624009;
        border: 1px solid rgba(255,255,255,0.78);
        border-radius: 16px;
        background: linear-gradient(145deg, #ffe9a6, #ffbd61);
        box-shadow: 0 12px 22px rgba(14,35,43,0.24), inset 0 1px 0 white;
        font-size: 1.25rem;
        transform: translateZ(55px) rotate(12deg);
    }

    [data-testid="stHorizontalBlock"] .st-key-step_focus,
    [data-testid="stHorizontalBlock"] .st-key-step_sources,
    [data-testid="stHorizontalBlock"] .st-key-step_irac,
    [data-testid="stHorizontalBlock"] .st-key-step_citations,
    [data-testid="stHorizontalBlock"] .st-key-step_journal {
        min-height: 84px;
        border: 1px solid #ffffff;
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(255,255,255,0.9), rgba(235,243,241,0.76));
        box-shadow: 0 9px 17px rgba(37,74,75,0.09), 0 2px 4px rgba(37,74,75,0.05);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }
    [data-testid="stHorizontalBlock"] .st-key-step_focus:hover,
    [data-testid="stHorizontalBlock"] .st-key-step_sources:hover,
    [data-testid="stHorizontalBlock"] .st-key-step_irac:hover,
    [data-testid="stHorizontalBlock"] .st-key-step_citations:hover,
    [data-testid="stHorizontalBlock"] .st-key-step_journal:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 24px rgba(37,74,75,0.13), 0 3px 6px rgba(37,74,75,0.07);
    }
    .st-key-step_focus { border-top: 3px solid #ed9460 !important; }
    .st-key-step_sources { border-top: 3px solid #32a99a !important; }
    .st-key-step_irac { border-top: 3px solid #6f76ca !important; }
    .st-key-step_citations { border-top: 3px solid #e2ae48 !important; }
    .st-key-step_journal { border-top: 3px solid #de7390 !important; }
    .st-key-input_card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-guide_card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-result_card [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(255,255,255,0.92);
        background: linear-gradient(148deg, rgba(255,255,255,0.96), rgba(244,248,245,0.91));
        box-shadow: 0 17px 33px rgba(34,71,74,0.11), 0 4px 9px rgba(34,71,74,0.06);
    }
    .st-key-input_card [data-testid="stVerticalBlockBorderWrapper"] { border-top: 4px solid #36a99d; }
    .st-key-guide_card [data-testid="stVerticalBlockBorderWrapper"] { border-top: 4px solid #8471cf; }
    .st-key-result_card [data-testid="stVerticalBlockBorderWrapper"] { border-top: 4px solid #e7ae4c; }
    .st-key-workflow_picker [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(255,255,255,0.92);
        background: linear-gradient(145deg, rgba(255,255,255,0.94), rgba(248,238,222,0.84));
        box-shadow: 0 10px 23px rgba(91,73,49,0.09);
    }
    [data-testid="stForm"] { padding: 0; border: 0; background: transparent; }
    [data-testid="stTextArea"] textarea {
        border-color: #d9e4df;
        background: #fcfdfa;
        line-height: 1.55;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #188b85;
        box-shadow: 0 0 0 3px rgba(24,139,133,0.15);
    }
    div[data-testid="stFormSubmitButton"] button {
        min-height: 48px;
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 14px;
        background: linear-gradient(110deg, #cf5d3f, #df7c44);
        box-shadow: 0 8px 15px rgba(180,76,49,0.24);
        font-weight: 700;
        transition: transform 180ms ease, box-shadow 180ms ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        border-color: rgba(255,255,255,0.56);
        background: linear-gradient(110deg, #b94c35, #cf6939);
        box-shadow: 0 12px 19px rgba(180,76,49,0.28);
        transform: translateY(-2px);
    }
    @keyframes rm-book-drift {
        0%, 100% { transform: translateY(0) rotateY(-21deg) rotateX(12deg); }
        50% { transform: translateY(-8px) rotateY(-17deg) rotateX(10deg); }
    }
    @media (max-width: 760px) {
        .stMainBlockContainer { padding: 1rem 1rem 2rem; }
        #research-mentor-hero { grid-template-columns: 1fr; min-height: 275px; padding: 28px; }
        #research-mentor-hero .rm-art { display: none; }
        #research-mentor-hero h1 { font-size: clamp(2.15rem, 8vw, 3rem); }
    }
    @media (prefers-reduced-motion: reduce) {
        #research-mentor-hero .rm-book-stack { animation: none; }
        [data-testid="stHorizontalBlock"] .st-key-step_focus,
        [data-testid="stHorizontalBlock"] .st-key-step_sources,
        [data-testid="stHorizontalBlock"] .st-key-step_irac,
        [data-testid="stHorizontalBlock"] .st-key-step_citations,
        [data-testid="stHorizontalBlock"] .st-key-step_journal { transition: none; }
    }
</style>
<section id="research-mentor-hero" aria-label="ResearchMentor introduction">
    <div class="rm-copy">
        <div class="rm-kicker"><span class="rm-kicker-mark"></span> YOUR LEGAL RESEARCH STUDIO</div>
        <h1>Big ideas.<br><span>Stronger arguments.</span></h1>
        <p>Turn a promising question into confident legal research with five focused AI specialists by your side.</p>
    </div>
    <div class="rm-art" aria-hidden="true">
        <div class="rm-rays"></div>
        <div class="rm-book-stack">
            <div class="rm-book book-back"></div>
            <div class="rm-book book-mid"></div>
            <div class="rm-book book-front">
                <span class="rm-book-title">§</span>
                <span class="rm-book-rule"></span>
                <span class="rm-book-rule short"></span>
                <span class="rm-book-rule"></span>
            </div>
            <div class="rm-float-chip">✦</div>
        </div>
    </div>
</section>
""")

with st.sidebar:
    st.header("Workspace settings")
    provider = st.selectbox(
        "AI provider",
        list(PROVIDERS),
        index=list(PROVIDERS).index(DEFAULT_PROVIDER),
        help="If one provider hits its free limit, switch to another here.",
    )
    provider_info = PROVIDERS[provider]
    model_label = st.selectbox("Model", list(provider_info["models"]))
    model_id = provider_info["models"][model_label]
    api_key = st.text_input(
        f"{provider.split(' (')[0]} API key",
        type="password",
        key=f"api_key_{provider}",
        help=f"Get a free key at {provider_info['key_url']}. "
        f"You can also set {provider_info['env']} in your environment.",
    )
    if api_key:
        st.caption(":green[API key entered]")
    else:
        st.caption(f":orange[Paste your key. Get one free: {provider_info['key_url']}]")
    if st.button("Test my API key", use_container_width=True):
        key_ok, key_message = test_api_key(provider, model_id, api_key)
        (st.success if key_ok else st.error)(key_message)
    st.caption(
        "Each run uses one specialist agent. Your text is sent to the chosen provider. "
        "Free tiers may use it to improve their products, so avoid confidential material."
    )
    with st.expander("About this project"):
        st.write(
            "ResearchMentor combines five CrewAI specialists for legal research support. "
            "It is an academic aid, not a substitute for checking primary legal sources "
            "or getting qualified legal advice."
        )

with st.container(border=True, key="workflow_picker"):
    st.subheader("Start with what you need")
    workflow_name = st.selectbox(
        "Choose a research task",
        options=list(WORKFLOWS),
        index=list(WORKFLOWS).index(
            st.session_state.get("active_workflow", list(WORKFLOWS)[0])
        ),
        key="workflow_select",
    )
    st.session_state["active_workflow"] = workflow_name

workflow = WORKFLOWS[workflow_name]

st.subheader("Your research journey")
steps = [
    ("Refine a research question", "step_focus", "01", "Focus", "lightbulb"),
    ("Discover legal sources", "step_sources", "02", "Find sources", "travel_explore"),
    ("Review an IRAC draft", "step_irac", "03", "Strengthen", "rule"),
    ("Format legal citations", "step_citations", "04", "Citations", "format_quote"),
    ("Prepare for journal submission", "step_journal", "05", "Publish", "publish"),
]
step_columns = st.columns(5, gap="small")
for column, (name, key, number, label, icon) in zip(step_columns, steps):
    with column:
        with st.container(border=True, key=key):
            is_active = workflow_name == name
            st.markdown(f":material/{icon}:  **{number}**")
            if is_active:
                st.badge(label, color="green")
            else:
                st.caption(label)

left, right = st.columns([1.35, 1], gap="large")
with left:
    with st.container(border=True, key="input_card"):
        st.subheader("Bring your material")
        st.caption("A question, a rough paragraph, or references you want to check.")
        with st.form("research_workflow", border=False):
            student_input = st.text_area(
                "Research question, draft, or citations",
                height=230,
                placeholder=workflow["placeholder"],
                help="Avoid including confidential or personally identifying information.",
            )
            run_workflow = st.form_submit_button(
                "Run this workflow",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )

with right:
    with st.container(border=True, key="guide_card"):
        st.subheader(f"{workflow['icon']} Your specialist")
        st.badge(workflow["specialist"], color="violet")
        st.markdown(f"**{workflow['description']}**")
        st.caption("YOUR RESULT")
        st.write(workflow["output"])
        with st.expander("Show an example prompt"):
            st.write(workflow["placeholder"])

if run_workflow:
    st.session_state.pop("workflow_result", None)
    if not student_input.strip():
        st.warning("Add a research question, draft, or citation before running this workflow.")
    elif not setup_environment(provider, api_key):
        st.error(
            "An API key is required for the chosen provider. Paste one in Workspace settings."
        )
    else:
        with st.spinner("Your specialist is working through the request..."):
            try:
                agent = workflow["agent"](get_llm(provider, model_id))
                task = Task(
                    description=workflow["task"](
                        normalize_model_input(student_input.strip())
                    ),
                    expected_output=workflow["expected"],
                    agent=agent,
                )
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    memory=False,
                    verbose=False,
                )
                result = str(crew.kickoff())
                st.session_state["workflow_result"] = {
                    "workflow": workflow_name,
                    "content": result,
                }
            except UnicodeEncodeError as error:
                st.error(
                    "This app process could not encode part of the request. "
                    "Restart the app with run_app.bat, refresh the page, and try again."
                )
                with st.expander("Technical details"):
                    st.code(str(error))
            except Exception as error:
                text = str(error).lower()
                if "429" in text or "rate limit" in text or "resource_exhausted" in text or "quota" in text:
                    st.warning(
                        "You have reached this provider's free limit for now. Wait a minute, "
                        "or pick a different provider or model in the sidebar and run again."
                    )
                else:
                    st.error(f"The workflow could not be completed: {error}")
                with st.expander("Technical details"):
                    st.exception(error)

if "workflow_result" in st.session_state:
    saved_result = st.session_state["workflow_result"]
    st.subheader("Your research feedback")
    st.caption(f"Workflow: {saved_result['workflow']}")
    with st.container(border=True, key="result_card"):
        st.markdown(saved_result["content"])
    st.download_button(
        "Download feedback",
        data=saved_result["content"],
        file_name="researchmentor-feedback.md",
        mime="text/markdown",
        icon=":material/download:",
    )

st.caption(
    "ResearchMentor is an academic support tool. Verify citations and legal propositions "
    "against authoritative sources."
)
