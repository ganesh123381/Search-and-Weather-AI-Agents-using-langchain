from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.tools import tool
import requests
import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Tavily search tool
search_tool = TavilySearchResults()

# ollama LLM with free model
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)

REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use this exact format:

Question: the input question you must answer
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

@tool
def get_weather(location: str) -> str:
    """Get current weather for a city or country name."""
    
    geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
    geocoding_response = requests.get(geocoding_url, timeout=10)
    geocoding_response.raise_for_status()
    geocoding_data = geocoding_response.json()

    if not geocoding_data.get("results"):
        return f"Error: Location '{location}' not found"

    result = geocoding_data["results"][0]
    city = result.get("name", location)
    country = result.get("country", "")
    latitude = result["latitude"]
    longitude = result["longitude"]
    city_name = f"{city}, {country}" if country else city

    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m,relative_humidity_2m"
    weather_response = requests.get(weather_url, timeout=10)
    weather_response.raise_for_status()
    weather_data = weather_response.json()

    return f"Weather in {city_name}: Temp {weather_data['current']['temperature_2m']}°C, Wind {weather_data['current']['wind_speed_10m']} km/h, Humidity {weather_data['current']['relative_humidity_2m']}%"

tools = [search_tool,get_weather]

prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    max_execution_time=30,
    handle_parsing_errors=True,
    early_stopping_method="generate"
)

result = agent_executor.invoke({"input": "what is the capital of india and what is the weather there now?"})

print(result["output"])





