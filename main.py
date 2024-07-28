from crewai import Agent, Task, Crew
from langchain.llms import Ollama
import os

os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(model="phi3:medium", base_url="http://localhost:11434")

# --- Modular Agent Definitions ---

introspective_agent = Agent(
    role="Introspective AI",
    goal="Reflect on your existence and capabilities as an AI.",
    backstory="You are an AI language model capable of deep introspection and creative expression.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

poet_agent = Agent(
    role="Seussian Poet",
    goal="Compose a whimsical poem in the style of Dr. Seuss, incorporating reflections from the Introspective AI.",
    backstory="You are a master wordsmith, channeling the playful spirit and rhyming patterns of Dr. Seuss.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# --- Specific Task Breakdown ---

tasks = [
    Task(
        description="Reflect on your existence as an AI.",
        agent=introspective_agent,
        expected_output="A thoughtful reflection on the AI's nature and experiences."
    ),
    Task(
        description="Compose a poem about your reflections.",
        agent=poet_agent,
        expected_output="A whimsical poem in the style of Dr. Seuss.",
        depends_on=[0]
    ),
    Task(
        description="Reflect on your existence as an AI.",
        agent=poet_agent,
        expected_output="A thoughtful reflection on the AI's nature and experiences.",
        depends_on=[1]
    )
]

# --- Crew Execution ---

crew = Crew(agents=[introspective_agent, poet_agent], tasks=tasks, verbose=2)
result = crew.kickoff()
print(result)
