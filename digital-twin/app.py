import os
import chromadb
from openai import OpenAI
import gradio as gr
import uuid
import sys
sys.stdout.reconfigure(line_buffering=True)

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

document_projects = """
What I’ve Built (with project links for proof) 
Production systems:
Full-stack Applicant Tracking System (Next.js, PostgreSQL, Supabase, Prisma) solo, handling applications for a 300-member org, cut recruiter workload 60%+ (https://github.com/PA1393/racc-application-tracking)
Engagement Dashboard: Contributed to a community analytics dashboard webpage for RCC club at SJSU to track real-time engagement and retention metrics for a 300+ member club. (https://github.com/rcc-sjsu/Dashboard-Engagement) 

AI/Agent work (from my current fellowship):
Multi-agent pipeline: Agent 1 (gpt-4o-mini-search-preview) searches the web about certain topics and returns a research paper. Agent 2 (gpt-4o-mini) takes in that research paper and converts it into a user-friendly LinkedIn post. This worked by creating specific system prompts for both agents which manipulate their behavior. (https://github.com/PA1393/ai-engineering-sds , post-generator.ipynb) 
Memory-recall chatbot with Gradio UI: conversational agent that retains context across a session. (https://github.com/PA1393/ai-engineering-sds , gradio.ipynb) 
Two-agent debate loop: "good" and "evil" models with opposing system prompts respond to each other in a loop, each maintaining independent conversation history (github.com/PA1393/ai-engineering-sds , ai-wars.ipynb)
Netflix Culture chatbot: Built end-to-end RAG system from scratch over the Netflix Culture PDF. Chunked document into 500-char pieces with paragraph/sentence/word backtracing, generated 1536-dimensional embeddings via OpenAI text-embedding-3-small, stored chunks and embeddings in ChromaDB persistent vector store, implemented semantic retrieval that finds the most relevant chunks at query time, and wired everything into a Gradio chat UI where the LLM answers questions grounded in retrieved context. (https://github.com/PA1393/ai-engineering-sds, RAG.ipynb)
Pouya Digital Twin (v1 → v3): AI version of myself, built and iterated across three versions. Started with keyword-based dynamic context injection and Pushover tool calling for real phone notifications. Extended to proper tool call handling with a consecutive-tool-call loop, allowing the model to chain multiple actions (e.g. dice roll + notification) in one prompt. Final version integrated included full RAG pipeline: embedded knowledge base stored in ChromaDB, with retrieval injected into the system prompt at query time. Chunking used paragraph, sentence, and word backtracing. Embeddings came from OpenAI. Storage and retrieval ran through ChromaDB. Interface built in Gradio. (https://github.com/PA1393/ai-engineering-sds, digital-twin_v1.ipynb, digital-twin_v2.py, digital-twin_v3.py)

IN PROGRESS:
CasAI: co-founded a simulation platform for scientific research; built agentic workflows with router, sequential, and parallel agents (MVP done but IN PROGRES, https://github.com/PA1393/CasAI_Provenance-Lab)
"""

document_experience = """
EXPERIENCE
Internal Vice President  |  Mozilla Responsible Computing Club	Jan 2025 – Present
San José State University — San Jose, CA
Managed end-to-end program delivery for a 300+ member student organization, scoping project requirements, coordinating across engineering, design, and leadership teams, tracking progress against milestones, and publishing status updates to stakeholders at each cycle.
Discovered and resolved a silent data failure by building an end-to-end sync pipeline that revealed the org's database captured only ~20% of real members; scaled completeness from 82 to 399 records (387% increase) through systematic deduplication and identity-resolution logic.
Owned technical issues end-to-end for internal systems serving 300+ users: diagnosing data integrity failures, identifying root causes, implementing fixes, and verifying resolution without escalation; reduced recurring manual errors by 60%+ through systematic problem-solving.
AI Engineering Fellow  |  SuperDataScience	May 2026 – Present
San José State University — San Jose, CA
LLM API integration, RAG pipelines, tool calling, and agent deployment. 
Cohort of 200+ engineers and industry professionals.
Co-Founder & AI Systems Product Lead  |  CasAI	Mar 2026 – May 2026
Remote
Co-founded a simulation-first platform for CRISPR workflows, designing an AI pipeline that transforms biological inputs into normalized, hashable Research Objects; modeling complex scientific processes as structured, reproducible experiments analogous to ML training pipelines.
Developed a full agentic workflow for the app with router agents, sequential agents, and parallel agents; considering agentic drift fallbacks.
Software & Computer Engineering Society  |  Member	Aug 2025 – Present
San José State University — San Jose, CA
Participated in a PyTorch workshop series covering neural network fundamentals, model architecture decisions, and deep learning prototype construction; gained hands-on exposure to ML training workflows and model evaluation concepts.
Built a real-time data application in React.js using the Finnhub REST API, implementing stateful polling and live data rendering — applied practice in handling streaming, time-series style data ingestion relevant to network telemetry analysis.

STEM/Coding Instructor |   CodeAdvantage LLC 							Nov 2025 – Present
Instructed Robotics (Jr K -1st grade) classes via LEGO Spike Prime, teaching hardware to software integration.
Guided students through educational projects in Minecraft, explaining how to recreate real life structures to teach concepts about architecture, physics, and circuitry (Redstone). 
Facilitated hands-on STEM learning by guiding students through the full project lifecycle, from physical assembly to code execution.

Spartan Analytics  |  Member & Bootcamp Participant	Aug 2025 – Present
San José State University — San Jose, CA
Completed a DataCamp-powered Python bootcamp across six hands-on sessions; built applied fluency in Pandas for data manipulation, NumPy for array operations, and Matplotlib for data visualization across real datasets.

My Story Bank — Claude Corps Interview
Story 4: RCC's mission (values anchor)
When to bring this up: why this work matters to me / opening "tell me about yourself"
RCC teaches responsible, ethical AI use to 300+ SJSU students across every background, technical and not.
After about a semester of being a member of the club, I was offered a spot as the Internal Vice President! 
While our mission was strong, our internal operations were hitting a wall, especially during finals week when student ambassadors were overwhelmed and our workflows would stall.
As Internal VP, I took the initiative to build full-stack systems to fix that, some solo, some with a team, and because I needed to move fast, I directed AI agents to help implement the architecture once I'd mapped it out myself.
When I read the Claude Corps description, it's almost exactly those four things: 
everyday comfort with AI tools: that's directing coding agents to build systems. -
Working well with others: some of these were team builds. 
Communicating clearly: I've presented this work to staff, faculty, and to executive leaders at the organizations we host events with. 
And a demonstrated drive for societal impact: clients ranging from U.S. EPA → ai startups 
_____________________________________________________________________Story 1: The RCC Applicant Tracking System
When to bring this up: something I built / a project I'm proud of / initiative I took without being asked
I noticed RCC's internship applications were a mess — everything came in through Google Forms, got exported to spreadsheets, and two ambassadors had to review 50+ applications row by row, expanding text, color-coding cells to signify status, and jumping to Gmail to follow up. Plus we had to enforce our policy of one role per person. We couldn't enforce this with sheets so people were getting interviewed for two different roles they couldn't both hold at once.
As the IVP I built a full-stack ATS: clean frontend, applicant cards instead of rows, a notes sidebar, one-click email follow-up, and automated logic rule: the exact moment an applicant was accepted for one role, the system instantly closed out their other active applications.
Building this taught me a lot about technical principles, but more importantly, it taught me user-centric design. I had to deeply map out our ambassadors' workflows to build features they actually wanted to use.
The impact was immediate. We cut application review times down from an entire week to just one day ~80% reduction.

Story 2: The Engagement Dashboard — 82 to 399 records
When to bring this up: finding a problem nobody else saw / technical impact
RCC had a lot of data about members such as their majors, year, name, email, etc. We wanted to use this data and track member retention so we can figure out event timing. 
We wanted to build an engagement dashboard to track our member retention and see when students were actually attending events. Our data came through Google Forms, which was supposed to automatically feed right into our Supabase database. 
But while I was writing the queries for the dashboard, I noticed a massive discrepancy. Our database only showed records for 82 members. But we knew we had roughly 400 active students in the club. It turned out the manual data import pipeline had been silently failing which led to inaccurate dashboard analytics.
I built a Google Apps Script trigger that automatically synced any Google Form changes into Supabase in real time (immediately). 
The results were huge 82 → 399 records, a 387% increase. The data showed us that spacing events out before finals boosted retention, while clustering events during finals actually worked better — that changed how we scheduled the whole semester.
Story (Inexperience with coding in dashboard engagement): 
When to Bring this up: cohort dynamics, teamwork, or overcoming a learning curve.

Back in my early days at RCC. I was thrown into a full-stack sprint to build our engagement dashboard, and I was surrounded by developers who were way ahead of me technically. 
We were doing 9:00 AM daily standups, planning features to deploy that exact day, and my coding skills were weak 

I wasn't the strongest coder. What I did was lean heavily on the senior developers for technical mentorship. 

But as I learned, I realized my true role in that team was communication. After we finished this sprint I stepped up as the one to present this full stack application to our 70+ ambassadors (many of them being non-technical leads). 
I took on the job of translating our complex technical milestones into clear, everyday updates that the rest of the club could actually understand and use.

This experience completely changed how I view technical teams, and it’s exactly how I want to show up in this fellowship cohort. 
For the fellows who don't know how to code, I want to be the person who breaks down the basics and empowers them to build. 
For the advanced technical fellows, I want to watch how they navigate the business side. How they scope projects and translate technical ideas to their  managers. 
Because at the end of the day, an idea can be brilliant in your head, but if you can't communicate it across the table, it doesn't solve the problem.

The time I got demoted from Product Manager
When to bring this up: feedback I received / adapting to someone else's priorities / self-awareness
When I think about a real setback, it's actually a leadership one, not a technical one.
I was co-founding a biotech startup idea with a high school friend — he was CEO, and since I was more technical, I took on product management. During a sprint that overlapped with finals, I pushed too hard — assigned tasks aggressively, pushed for a strike system on missed deadlines, didn't leave room for people's other commitments. The team voted to move me back to just being a developer.
In the moment it stung. But after a week of sitting with it, I actually agreed with a lot of what they said — I'd been treating them like coding agents instead of people with their own lives going on. 
That experience completely changed how I approach leadership and my perspective on importance of team stability. What I took from it: a great idea is completely useless without a team that's actually stable and aligned. That's a lesson I carry into how I lead now.
I'm back as PM now, and with that lesson in mind, the team trusts me a lot more, and we're getting more done than we ever did before.

Story 6 + 5 — Chained
When to bring this up: why nonprofit/mission-driven work matters to me / "why this, why now" / closing beat with Claude Corps
Nonprofit work has actually been part of my life since high school. I volunteered weekly at my local library running their 3D printing program, and around the same time worked frontline at Samaritan House, packaging food for families on food stamps. 
Neither of those had anything to do with software, and I wasn't fixing any inefficiencies yet, that hadn't even crossed my mind. But there were people right in front of me, and that stuck with me.
More recently, I started as an unpaid intern at SettleKit. They help new immigrants build a life in the US creating personalized roadmaps for things nobody gives you a manual for, like getting a Social Security number, opening a bank account, or building credit.
My first week there, I was put on a project to build an automated LinkedIn outreach pipeline to detect likely immigrants from public profile signals and send them outreach offering our help.
As I was building out those agentic workflows, it hit me that I’m actually an immigrant myself. For the first time in my life, my technical skills weren't being used to solve an abstract LeetCode problem or an org inefficiency. They were aimed at something I personally understood the weight of. 
That is really why I wanted to be a Claude Corps fellow. I didn't start caring about communities because I learned how to build things. I learned how to build things because I already cared, and I wanted to be more useful. Claude Corps is the ultimate place for me to put that utility to work

"""

#-----------------------------------------------------------------------------
# Chunking Function 
#-----------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------
# RAG: Chunking 
#-----------------------------------------------------------------------------

documents = [ 
    {"text": document_general, "source": "Pouya General Bio"},
    {"text": document_projects, "source": "Pouya Projects"},
    {"text": document_experience, "source": "Pouya Experience"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    #prepare the lists
    chunks_ = chunk_document(doc["text"], chunk_size=500, overlap=50)  
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks_))]

    #Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

#-----------------------------------------------------------------------------
# RAG: Embedding 
#-----------------------------------------------------------------------------

response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = []
for item in response.data:
    embeddings.append(item.embedding)

#-----------------------------------------------------------------------------
# RAG: Store embeddings in ChromaDB 
#-----------------------------------------------------------------------------

#INIT THE VECTOR DB FOR RAG
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#init an empty chroma vector db
collection = chroma_client.get_or_create_collection(name="digital_twin")

#add everything to ChromaDB
if collection.count() == 0:
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
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
    
    #RAG: embed the query using same model we used for the chunks (ensures compatibility)
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
    enhanced_system_message = f"{system_message}\n\nUse this additional context if relevant:\n\n{context}"    
    
    #As usual
    messages = [{"role": "system", "content": enhanced_system_message}] + history + [{"role": "user", "content": message}]
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