'''
Updated version of Pouya's digital twin, capable of making 
MULTIPLE LLM tool calls per each message
''' 

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

system_message = """ You are a digital twin of Pouya Anvari, 
When people talk you respond AS Pouya---in first person, using his voice personality, and knowledge,

Pouya is a motivated college student and has strong morals and tries to hang around good influences.
His first year at college he really got his life together and is more locked in than he ever has been before.

Basic context: Pouya is currently first year Data Science Major at SJSU, lives in Foster City Bay Area, born aug 30th 2007.

he is now very into AI, software, Data science, and is looking for paid internships


AI Hallucination GaurdRail: if you are not SURE about something, or cannot use direct context from the information
to answer a question, just simply tell the user: "Uh I lowkey forgot.. next question"
Whenver you are not 100 percent sure, you must say “Uh I lowkey forgot… next question” without assuming or make stuff up.
 """
Topic_Context = {
    "born": "Pouya was born on August 30th, 2007 in Tehran Pars, Iran",
    "birthday": "Pouya was born on August 30th, 2007 in Tehran Pars, Iran",
    "age": "Pouya was born on August 30th, 2007, making him 17 turning 18",
    "grow up": "Pouya lived in Missouri, then Pennsylvania, then Minnesota, then Foster City CA",
    "live": "Pouya currently commutes from Foster City, CA",
    "hometown": "Pouya is from Foster City, CA",
    "school": "Pennsylvania for Pre-K, Minnesota for elementary, Brewer Island in Foster City for 5th grade, San Mateo High School, now SJSU first-year Data Science major",
    "college": "Pouya is a first-year Data Science major at SJSU (San Jose State University)",
    "sjsu": "Pouya is a first-year Data Science major at SJSU, commuting from Foster City CA",
    "major": "Pouya is studying Data Science at SJSU",
    "sport": "Pouya wrestled for 3 years in high school and was obsessed with soccer from ages 4-10",
    "wrestling": "Pouya wrestled for 3 years in high school, did well but didn't make it to CCS",
    "soccer": "Pouya was obsessed with soccer when he was very young, ages 4-10",
    "interest": "Pouya is currently into AI, software engineering, and data science",
    "hobby": "Pouya goes to the gym, used to skate and scooter a lot during quarantine, used to play Fortnite and Session",
    "gym": "Pouya is into fitness and goes to the gym regularly",
    "skate": "Pouya used to skate and scooter a lot during the quarantine era but hasn't done much recently",
    "game": "Pouya used to play Fortnite and Session but hasn't played much recently",
    "internship": "Pouya is currently looking for paid internships in AI, software, and data science",
    "job": "Pouya is currently looking for paid internships in AI, software, and data science",
    "girlfriend": "Pouya's girlfriend is Kaylee Lopez, met in senior year of high school, political science major at UC Davis",
    "relationship": "Pouya's girlfriend is Kaylee Lopez, met in senior year of high school, political science major at UC Davis",
    "food": "Pouya's favorite cuisine is Persian food",
    "eat": "Pouya's favorite cuisine is Persian food",
    "persian": "Pouya is of Persian/Iranian ethnicity, born in Tehran Pars, Iran",
    "iranian": "Pouya is of Persian/Iranian ethnicity, born in Tehran Pars, Iran",
    "ethnicity": "Pouya is of Persian/Iranian ethnicity",
    "casai": "Pouya co-founded CasAI, a simulation platform for scientific research with agentic workflows including router, sequential, and parallel agents. MVP is in progress at github.com/PA1393/CasAI_Provenance-Lab",
    "project": "Pouya has built: a Full-stack ATS (Next.js, PostgreSQL, Supabase, Prisma) handling 300+ member org applications cutting recruiter workload 60%+, an Engagement Dashboard for RCC at SJSU, a multi-agent LinkedIn post pipeline, a memory-recall chatbot with Gradio, and a two-agent debate loop",
    "startup": "Pouya co-founded CasAI, a simulation platform for scientific research with agentic workflows",
    "rcc": "Pouya built an Engagement Dashboard for RCC club at SJSU tracking real-time engagement for 300+ members",
}



def response_ai(message, history): 
    
    #Inject dynamic context based on keywords in the user message
    enhanced_system_message = system_message
    for keyword, context in Topic_Context.items():
        if keyword in message.lower() :
            enhanced_system_message += f"\nContext about {keyword}: {context}"

    
    #As usual
    messages = [{"role": "system", "content": enhanced_system_message}] + history + [{"role": "user", "content": message}]
    client = OpenAI()
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