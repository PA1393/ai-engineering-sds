import os
from openai import OpenAI
import gradio as gr

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API key missing")
client = OpenAI()

#-----------------------------------------------------------------------------
# Document 
#-----------------------------------------------------------------------------

document_general = """ 

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

#-----------------------------------------------------------------------------
# System Message
#-----------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------
# Main Response Function
#-----------------------------------------------------------------------------
def response_ai(message, history): 
    #updated system message with context
    enhanced_system_message = f"{system_message}\n\nUse this additional context if relevant:\n\n{document_general}"    

    #Logs
    print("\n================================\n")
    print(f"User Message: {message}")
    print("Context this turn:\n", enhanced_system_message)

    #Build messages for this turn
    messages = [{"role": "system", "content": enhanced_system_message}] + history + [{"role": "user", "content": message}]

    #Call LLM 
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    message = response.choices[0].message
    
    return message.content
  

#-----------------------------------------------------------------------------
# Launch Gradio
#-----------------------------------------------------------------------------
demo = gr.ChatInterface(fn=response_ai)
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))