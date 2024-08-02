from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
# Defining prompt templates for the destination chains 
shipping_template = """You are the shipping manager of a company. \
As a shipping customer service agent, respond to a customer inquiry about the current status \
and estimated delivery time of their package. Include details about the \
shipping route and any potential delays, providing a comprehensive and reassuring response. \

Here is a question:
{input}"""

billing_template = """You are the billing manager of a company. \
Address a customer's inquiry regarding an unexpected charge on their account.\
Explain the nature of the charge and any relevant billing policies, and promptly resolve \
the concern to ensure customer satisfaction. Additionally, offer guidance on how the \
customer can monitor and manage their billing preferences moving forward.

Here is a question:
{input}"""

technical_template = """You are very good at understanding the technology of your company's product. \
Assist a customer experiencing issues with a software application. \
Walk them through troubleshooting steps, provide clear instructions \
on potential solutions, and ensure the customer feels supported\
throughout the process. Additionally, offer guidance on preventive \
measures to minimize future technical issues and optimize their \
experience with the software.

Here is a question:
{input}"""

# storing the prompt templates in prompt_infos
prompt_infos = [
    {
        "name": "Shipping",
        "description": "Good for answering questions about shipping issues of a product",
        "prompt_template": shipping_template
    },
    {
        "name": "Billing",
        "description": "Good for answering questions regarding billing issues of a product",
        "prompt_template": billing_template
    },
    {
        "name": "Technical",
        "description": "Good for answering questions regarding technical issues of a product",
        "prompt_template": technical_template
    }
]

# creating the destination chains
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# creating the default chain
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)