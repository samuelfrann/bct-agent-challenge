# 🇳🇬 Naija-Aware Multi-Agent Orchestration

**A Decoupled Microservice Architecture for High-Fidelity User Modeling and Contextual Retrieval-Augmented Generation.**

Built for the **BCT x DSN LLM Agent Challenge**, this project abandons fragile monolithic LLM prompt chains in favor of a robust, container-ready microservice topology. It solves complex spatial routing, abstract user intent, and the "cold-start" anomaly, all while natively infusing Nigerian socio-cultural realities into the foundation model's reasoning space.

---

## 🏗️ System Architecture

Our solution is divided into three decoupled layers to ensure UI stability, graceful payload degradation, and strict separation of concerns:

1. **Client Layer (React / Vite):** Captures abstract user intents and renders agentic Chain-of-Thought (CoT) traces without blocking UI threads.
2. **API Gateway (Node.js / Express):** An intelligent hydration middleware that intercepts frontend requests, maps them to strict Pydantic schemas, and manages connection timeout fallbacks.
3. **Inference Engine (FastAPI / LangGraph):** A deterministic Python backend orchestrating state machines, utilizing local ChromaDB vector spaces and Anthropic's Claude 3.5 Sonnet.

---

## 🚀 Key Features

### Task A: Predictive User Modeling (Review Simulator)
* **Dynamic Semantic Anchoring:** Uses cosine similarity to pull a user's past reviews of *similar* venues from ChromaDB, injecting them as few-shot RAG examples to force the LLM to adopt the user's specific vocabulary and variance.
* **Dual-Objective Optimization:** Minimizes quantitative error while maximizing qualitative semantic fidelity.

### Task B: Context-Aware Recommendation Engine
* **Retrieve-then-Rerank Pipeline:** Filters top candidate venues locally via ChromaDB (saving token limits), then uses the LLM as a semantic reranker to evaluate soft attributes (e.g., "anniversary viability").
* **Autonomous Cold-Start Routing:** Dynamically routes new users without historical data to a semantic proxy search based solely on their Gateway-injected persona description.

### The "Naija Layer" (Cultural Context Module)
A parameterized engine that structuralizes true Nigerian localization:
* **Infrastructural Reality:** LLM penalizes venues for frequent generator noise or faulty ACs during humid seasons.
* **Spatial Friction:** Mathematically penalizes cross-bridge routing attempts (e.g., Mainland to Island) during peak traffic hours.
* **Linguistic Parameterization:** Blends standard Nigerian English with balanced Pidgin syntax, adapting tone based on the city (Lagos vs. Ibadan vs. Port Harcourt).

---

## ⚙️ Prerequisites

Before you begin, ensure you have met the following requirements:
* **Node.js** (v18.0.0 or higher)
* **Python** (v3.10 or higher)
* **Anthropic API Key** (Claude 3.5 Sonnet access)

---

## 🛠️ Installation & Setup

Because this is a microservice architecture, you will need to open **three separate terminal windows** to run the local stack.

### 1. Backend Inference Engine (Python/FastAPI)
```bash
# Navigate to the root directory
cd bct-agent-challenge

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Create a .env file and add your Anthropic API Key
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env

# Run the local data ingestion script to populate ChromaDB
python retrieval/ingest.py

# Boot the FastAPI server
python main.py

2. API Gateway (Node.js/Express)

# Open a second terminal and navigate to the gateway folder
cd bct-agent-challenge/gateway

# Install dependencies
npm install express cors dotenv axios

# Boot the Gateway
node server.js

3. Frontend Client (React/Vite)

# Open a third terminal and navigate to the frontend folder
cd bct-agent-challenge/frontend

# Install dependencies
npm install

# Start the development server
npm run dev

💻 Usage
Open http://localhost:5173 in your browser.

Navigate to Task A to test the User Modeling Agent. Input a persona (e.g., "Generous rater who loves detailed write-ups") and context (e.g., "Premium coastal dining").

Navigate to Task B to test the Recommender. Input abstract intents (e.g., "Outdoor spot in VI for a first date") and watch the agent output culturally reasoned rankings.

📁 Repository Structure
Plaintext
├── agents/                 # LangGraph state machines (task_simulator.py, task_recommender.py)
├── core/                   # Core parsing and context engines (naija_layer.py, persona_parser.py)
├── data/                   # JSONL slices for ChromaDB ingestion
├── frontend/               # React UI components (App.jsx)
├── gateway/                # Node.js Express server (server.js)
├── retrieval/              # Vector database logic and ingestion scripts
├── schemas/                # Pydantic data models
├── main.py                 # FastAPI application entry point
└── README.md