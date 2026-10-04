from crewai import Agent
from tools.research_tools import verify_reference_crossref

def get_citation_integrator(llm=None) -> Agent:
    return Agent(
        role="OSCOLA & Bluebook Legal Citation Verification Specialist",
        goal="Convert messy citations into precise OSCOLA/Bluebook footnote entries.",
        backstory=(
            "You are a legal citation accuracy expert. You ensure every statute, report (PLD, SCMR, CLC, PCrLJ, YLR, MLD, PLC), "
            "and journal entry complies strictly with OSCOLA and Bluebook rules."
        ),
        tools=[verify_reference_crossref],
        verbose=True,
        llm=llm,
    )
