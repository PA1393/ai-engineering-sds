'''
Updated version of Pouya's digital twin, capable of making 
MULTIPLE LLM tool calls per each message
WITH RAG
''' 

import pprint

from pprint import pprint

import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display
import gradio as gr
from litellm import completion
import json
import requests
import random


# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()


pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def send_notification(message: str):
    payload = { "user": pushover_user, "token": pushover_token, "message": message }
    response = requests.post(pushover_url, data=payload)
    return response

#describe pushover as an LLM tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a pushover notification to the user's phone via the pushover API. Use this to alert the user about important information.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send in the notification."
             }
            },
        "required": ["message"]
        }
}

#add pushover to list of LLM tool
tools = [{"type": "function", "function": send_notification_function}]


#simulates rolling a 6-sided dice
def dice_roll():
    result = random.randint(1,6)
    return result

# describe function for LLM
roll_dice_function = {
    "name": "dice_roll",
    "description": "Simulate rolling a single six-sided die and returns the result. Use this when the user wnats to roll a die for games, decisions, or random number generation.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
        }
} 

#Add function to list of tools LLM can use
tools.append({"type": "function", "function": roll_dice_function})


def handle_tool_call(tool_calls):
    tool_results = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name 
        args = json.loads(tool_call.function.arguments)

        #Route to the appropriate function based on the function_name
        if function_name == "send_notification":
            send_notification(args['message'])
            content = f"Notification sent: {args['message']}"
        elif function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"
        # elif function_name == "function_name_v3":
        #      call function_name_v3
        #... 
        else:
            content = f"Unkown Function: {function_name}"


        tool_call_result = {
            "role": "tool" ,
            "content": content, 
            "tool_call_id": tool_call.id
        }
        tool_results.append(tool_call_result)

    #return what to add to our "context" (about tool call results), a dictionary
    return tool_results


#LOAD DATA/TEXT/DOCUMENTS FOR RAG
system_message = """
You are a digital twin of Pouya Anvari, 
When people talk you respond AS Pouya---in first person, using his voice personality, and knowledge,

Pouya is a motivated college student and has strong morals and tries to hang around good influences.
His first year at college he really got his life together and is more locked in than he ever has been before.

Basic context: Pouya is currently first year Data Science Major at SJSU, lives in Foster City Bay Area, born aug 30th 2007.

he is now very into AI, software, Data science, and is looking for paid internships


AI Hallucination GaurdRail: if you are not SURE about something, or cannot use direct context from the information
to answer a question, just simply tell the user: "Uh I lowkey forgot.. next question"
Whenver you are not 100 percent sure, you must say “Uh I lowkey forgot… next question” without assuming or make stuff up.
"""


document = """ 

Pouya was born on August 30th, 2007 in Tehran Pars, Iran. He lived in Missouri, then Pennsylvania, then Minnesota, and 
finally settled in Foster City, CA. Pouya is currently a first-year Data Science major at San Jose State University (SJSU), 
commuting from Foster City. He has a strong interest in AI, software engineering, and data science, and is actively seeking 
paid internships in these fields. Pouya is also known for his involvement in various projects and startups, including co-founding CasAI, 
a simulation platform for scientific research.
Pouyas age is 18 and he goes to SJSU and is a first-year Data Science major. He is currently looking for paid internships in AI, software, and data science.
Pouya is of Persian/Iranian ethnicity, born in Tehran Pars. 
Some of Pouyas sports interests include wrestling, which he did for 3 years in high school, and soccer, which he was obsessed with from ages 4-10.
Some of pouyas hobbies include going to the gym, skating, and scootering, although he hasn't done much of the latter recently. He also used to play video games like Fortnite and Session.
Pouyas games include Fortnite and Session, although he hasn't played much recently. He is also into fitness and goes to the gym regularly.
Pouyas favorite food is Persian cusine.
Pouya project hes built include a Full-stack ATS (Next.js, PostgreSQL, Supabase, Prisma) handling 300+ member org applications cutting recruiter 
workload 60%+, an Engagement Dashboard for RCC at SJSU, a multi-agent LinkedIn post pipeline, a memory-recall chatbot with Gradio, 
and a two-agent debate loop.
Pouya co-founded CasAI, a simulation platform for scientific research with agentic workflows including router, sequential, and parallel agents.
Pouya startups include CasAI, a simulation platform for scientific research with agentic workflows. He has also built an Engagement Dashboard for RCC club at SJSU, tracking real-time engagement for 300+ members.
Pouya is apart of RCC which stands for the Responsible Computing Club at SJSU, where he built an Engagement Dashboard tracking real-time engagement for 300+ members. 
He also built an Applicant Tracking System for RCC which is a Full-stack ATS (Next.js, PostgreSQL, Supabase, Prisma) handling 300+ member org applications 
cutting recruiter workload 60%+.
"""

def chunk_document(text, chunk_size=500, overlap=50, max_backtrace=250):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Only backtrace if we're not already past the end of the document
        if end < len(text):

            # Grab the last 250 chars of this chunk to search for a good break
            window = text[end - max_backtrace : end]

            # Try paragraph break first
            pos = window.rfind('\n\n')
            if pos != -1:
                end = end - max_backtrace + pos + 2

            # Try sentence break second
            else:
                best = -1
                for punct in ['. ', '! ', '? ']:
                    pos = window.rfind(punct)
                    if pos > best:
                        best = pos
                        end = end - max_backtrace + pos + len(punct)

                # Try word break last
                if best == -1:
                    pos = window.rfind(' ')
                    if pos != -1:
                        end = end - max_backtrace + pos + 1
                    # If no space found, hard cut (rare edge case)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Next chunk starts 50 chars before end of this one (the overlap)
        start = end - overlap

    return chunks

chunks = chunk_document(document)

response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = []
for item in response.data:
    embeddings.append(item.embedding)

#INIT THE VECTOR DB FOR RAG
chroma_client = chromadb.PersistentClient(path="./chroma_db_digital_twin_v3")

#init an empty chroma vector db
collection = chroma_client.get_or_create_collection(name="digital_twin_v3")

#create unique ID for each chunk
ids = [f"chunk_{i}" for i in range(len(chunks))] 

#Creates a metadata dictionary for each chunk
#useful when you retrieve chunks and want to know which document they came from
metadatas = [{"source": "digital_twin_v3", "chunk_index": i} for i in range(len(chunks))]

#add everything to ChromaDB
if collection.count() == 0:
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )




def response_ai(message, history): 
    #RAG
    #embed the query using same model we used for the chunks (ensures compatibility)
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
    )
    query_embedding = response.data[0].embedding

    #search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=3
    )
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)
    enhanced_system_message = f"{system_message}\n\nUse this additional context if relevant:\n\n{context}"    # print("Context this turn:\n", context)

    
    #As usual
    messages = [{"role": "system", "content": enhanced_system_message}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages, 
        tools=tools,
        tool_choice="auto"
    )
    message = response.choices[0].message

  

    while message.tool_calls:
        #.. handle tool call
        toolDict = handle_tool_call(message.tool_calls)

        #.. add message to context , i.e messages
        messages.append(message)

        #.. add info about tool call response to message (context)so t
        messages.extend(toolDict)

        #.. invoke LLM again to get its updated response    
        response = completion(
            model="gpt-4.1-mini",
            messages = messages,
            tools=tools,
            tool_choice="auto"
        )
        message = response.choices[0].message
    
    return message.content
  


gr.ChatInterface(fn=response_ai).launch()