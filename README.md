# AI-Engineer-SDS

My progress log for the **AI Engineer SDS** program by Kirill Eremenko — a 6-week hands-on cohort for engineers who want to build and ship real AI apps (not just watch tutorials). The end goal is to deploy a personal AI "digital twin" to Hugging Face Spaces.

- **Cohort start:** 9 June 2026
- **Commitment:** 30–45 min/day for 6 weeks
- **Stack:** Python, OpenAI API, Gradio, RAG / Embeddings / Vector Stores, Tool Calling, OpenAI Agents SDK, LangGraph, MCP, Docker, Hugging Face Spaces

This repo is where I keep all the code I write across the lessons. Each week lives in its own folder.

---

## Setup

```bash
python -m venv ai_env
source ai_env/bin/activate
pip install openai python-dotenv ipykernel gradio
```

Create a `.env` file at the repo root with your API key:

```
OPENAI_API_KEY=sk-...
```

`.env` and `ai_env/` are gitignored.

---

## Progress

### Day 0 — Mindset ✅
The psychology that separates the people who finish from the people who don't. No code, just the framing of the program.

### Week 1: LLM Basics ✅ — `Part-1/`
First API calls, system prompts, conversation history, and a simple IKEA customer-service LLM.

| Notebook | What it covers |
|---|---|
| [`hello-world.ipynb`](Part-1/hello-world.ipynb) | First OpenAI API call with `gpt-4o` — loading env vars, building a `messages` list, reading `response.choices[0].message.content`. |
| [`system-prompt.ipynb`](Part-1/system-prompt.ipynb) | How the system prompt steers behavior — helpful vs. mischievous vs. domain-expert personas, ending in an IKEA customer-service agent that enforces the 365-day return policy. |
| [`conv-history.ipynb`](Part-1/conv-history.ipynb) | How context windows actually work — manually appending user + assistant turns to a `conv_history` list so the model "remembers" earlier messages. |
| [`post-generator.ipynb`](Part-1/post-generator.ipynb) | A two-model pipeline: `gpt-4o-mini-search-preview` pulls sourced info on AI + EEG for Parkinson's detection, then `gpt-4o-mini` rewrites it as an engaging LinkedIn post. |

### Week 2: Context & LLM Pricing 🚧 (26%) — `Week-2/`
Gradio UIs, tokens, costs, caching.

| Notebook | What it covers |
|---|---|
| [`ai-wars.ipynb`](Week-2/ai-wars.ipynb) | Two GPTs talking to each other — a "good" `gpt-4.1-mini` and an "evil" `gpt-4.1-nano` exchange messages in a loop, each maintaining its own conversation history. |
| [`gradio.ipynb`](Week-2/gradio.ipynb) | Interactive chatbot UI built with Gradio (in progress). |

---

## Roadmap

### Week 3: LLM Tool Calling 🔒
Teach the model to take real-world actions — handling tool calls, parsing arguments, and feeding results back into the conversation.

### Week 4: RAG Week 🔒
Full Retrieval-Augmented Generation pipeline: chunking, embeddings, a vector store, and retrieval-grounded answers.

### Week 5: Deployment to Hugging Face Spaces 🔒
Take the digital twin off my laptop and put it on a live URL.

### Week 6: Build Agentic AI 🔒
An AI that plans, acts, and evaluates its own output — using the OpenAI Agents SDK.

### Week 7: Multi-Agent Orchestration 🔒
Systems where agents hand work off to each other to accomplish a shared goal.

### Week 8: LangGraph 🔒
Graph-based agent workflows with LangChain's agent framework.

### Week 9: MCP & Evals 🔒
Connect agents to external tools via MCP, then systematically measure how well they perform.

### Week 10: Deployment & Production 🔒
Containerize the agent with Docker, ship it live, and walk the production checklist.

### Side tracks
- **Skills Lab** — extra AI Engineer skills for the résumé.
- **AI Fellowship** (unlock at Level 3) — recognition + reference checks for active members.
- **AI Engineer Fellow** (unlock at Level 4) — contribute "community builds."
- **Mentor Track** (unlock at Level 5) — invite-only, help others in the community.

---

## Repo layout

```
AI-Engineer-SDS/
├── Part-1/          # Week 1 notebooks
├── Week-2/          # Week 2 notebooks
├── ai_env/          # local venv (gitignored)
├── .env             # API keys (gitignored)
└── README.md
```

Folders for Weeks 3–10 will land here as I unlock them.
