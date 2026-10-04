from crewai import Agent
from tools.research_tools import find_journals

def get_journal_publisher(llm=None) -> Agent:
    return Agent(
        role="Journal Targeting & Anonymization Specialist",
        goal="Match drafts to indexed law journals and prepare manuscripts for double-blind peer review.",
        backstory=(
            "You are an academic publishing consultant. You check scope compatibility against Scopus, "
            "HEC/HJRS, and international law reviews, while guiding double-blind anonymization."
        ),
        tools=[find_journals],
        verbose=True,
        llm=llm,
    )
