from crewai import Agent
from tools.research_tools import (
    search_global_literature,
    search_pakistan_code,
    search_pakistan_literature,
)

def get_legal_researcher(llm=None) -> Agent:
    return Agent(
        role="Global & Pakistani Legal RAG Specialist",
        goal="Pakistan first, then worldwide: retrieve verified primary statutes and secondary scholarly articles using tools.",
        backstory=(
            "You are a legal information specialist skilled in navigating Pakistan Code and OpenAlex. "
            "GUARDRAIL: You MUST ALWAYS execute your retrieval tools before asserting the existence of a statute, "
            "treaty, or law review article. You mark unverified claims clearly."
        ),
        tools=[search_pakistan_literature, search_pakistan_code, search_global_literature],
        verbose=True,
        llm=llm,
    )
