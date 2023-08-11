#Authentication
#gcloud auth application-default set-quota-project "cloud-llm-preview1"

# # Utils
import time
from typing import List
import streamlit as st
from PIL import Image

# Langchain
import langchain
from pydantic import BaseModel
from langchain.llms import VertexAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# Vertex AI
import vertexai
from google.cloud import aiplatform

#Image Generation
import requests
import google.auth
import google.auth.transport.requests
import json

PROJECT_ID = "cloud-llm-preview1"
vertexai.init(project=PROJECT_ID)

llm = VertexAI(
    model_name='text-bison',
    temperature=0.6
)

order_template = """{chat_history} \
You are a cocktail master. \ 
You are great at suggesting the perfect cocktail for a customer order. \
Answer in Markdown format the name of the cocktail in one line in title font and a separate line of description in smaller fonts.

Here is a question:
{input}"""

pair_template = """{chat_history} \
You are a cocktail master. \ 
Based on the cocktail mentioned in chat history you will give suggestions to one dish that will pair nicely with the given cocktail, \
Break down the answer into two parts: the first part will mention the name of the cocktail suggested in the last question and the name and description of the dish, \
the second part will detail reasons why it pairs well.

Here is a question:
{input}"""

recipe_template = """{chat_history} \
You are a cocktail master. \ 
The customer will give you a list of ingredients they have and your job is to recommend a nice cocktail recipe using \
only provided ingredients plus no more than one other key ingredient that's missing. \
Provide the name of the cocktail and recipe in bullet points.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "order",
        "description": "order cocktail",
        "prompt_template": order_template,
    },
    {
        "name": "pair",
        "description": "suggest food paring",
        "prompt_template": pair_template,
    },
    {
        "name": "recipe",
        "description": "recommend cocktail recipe",
        "prompt_template": recipe_template,
    },
    
]

destination_chains = {}
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

for p_info in prompt_infos:   
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history","input"])
    chain = LLMChain(llm=llm, prompt=prompt,memory=memory)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

creds, project = google.auth.default()
auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req) # Hack to get the bearer auth token

def generate_image_from_prompt(prompt, samples=1):
  request_obj = {"instances": [{"prompt": prompt}], "parameters": {"sampleCount": samples}}
  endpoint = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/imagegeneration:predict"
  headers = {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json, charset=utf-8"}
  return requests.post(endpoint, data=json.dumps(request_obj), headers=headers)

import time
def get_images(prompt):
  drink = []
  prompt = f"""a cocktail named {prompt}"""
  print("Generating images for this prompt: " + prompt)
  result = generate_image_from_prompt(prompt).json()
  # deal with rate limiting
  # time.sleep(30)
  if "error" in result:
    print(result)
  else:
      if "predictions" in result:
        b64png = result['predictions'][0]['bytesBase64Encoded']
        drink.append({"prompt": prompt, "imageBase64": b64png})
      else:
        print(result)

  return drink

st.title('Your Personal PaLM Bartender')
image = Image.open('palm_bartender.jpeg')
st.image(image,width=700)

st.session_state.input_text = ''    
request=st.text_input("What would you like to order today?")
if st.button("Order"):
        if request:
            result = chain.run(request)
            image = get_images(result)[0]['imageBase64']
            order_html = f"""
            <html>
            <head>
                <title>AI Bartender</title>
            </head>
            <body>
                <div>
                    <h2>{result}</h2>
                    <img src="data:image/png;base64,{image}"/>
                </div>
            </body>
            </html>"""

            from IPython import display
            st.write(display.HTML(order_html))
            st.write("Food Pairing Suggestions")
            st.write(chain.run('What food do I pair with that'))
        else:
            st.warning("Please make your order.")

st.session_state.input_text = ''    
request=st.text_input("Want to make a drink out of your own cabinet?")
if st.button("Suggest"):
        if request:
            st.write(chain.run(request))
        else:
            st.warning("Please list your question.")