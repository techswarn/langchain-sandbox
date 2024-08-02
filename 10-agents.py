import os
# importing LangChain modules
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

os.environ["SERPAPI_API_KEY"] = "{{SERPAPI_API_KEY}}"

# Insert your key here
llm = OpenAI(temperature=0.0,
            openai_api_key = "{{OpenAI_Key}}")

# loading tools
tools = load_tools(["serpapi", 
                    "llm-math"], 
                    llm=llm)

agent = initialize_agent(tools, 
                        llm, 
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True)

# user's query
print(agent.run("What is the current population of the world, and calculate the percentage change compared to the population five years ago"))