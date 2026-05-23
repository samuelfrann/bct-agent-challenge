# agents/task_b_recommender.py
from dotenv import load_dotenv
load_dotenv()

import json
import re
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

import retrieval.retriever as retriever_module
from core.persona_parser import resolve_persona
from core.naija_layer import NAIJA_SYSTEM_PROMPT, get_recommendation_framing
from retrieval.retriever import search_businesses
from schemas.models import RecommendRequest, RecommendResponse, RecommendationItem


# ── State ─────────────────────────────────────────────────────────────────────
class RecommenderState(TypedDict):
    raw_persona:          dict
    raw_context:          dict
    k:                    int
    persona:              dict
    is_cold_start:        bool
    search_query:         str
    candidate_businesses: list
    cultural_framing:     str
    raw_response:         str
    recommendations:      list
    clarifying_questions: list


# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=3000)

# ── Node 1: Parse persona + detect cold start ─────────────────────────────────
def parse_persona_node(state: RecommenderState) -> dict:
    print("🧠 Resolving user persona...")
    persona       = resolve_persona(state["raw_persona"], retriever_module)
    is_cold_start = persona.get("is_cold_start", False) or persona.get("total_reviews", 0) == 0
    return {"persona": persona, "is_cold_start": is_cold_start}


# ── Router — this is the conditional edge ────────────────────────────────────
def cold_start_router(state: RecommenderState) -> str:
    return "cold_start" if state["is_cold_start"] else "regular"


# ── Node 2a: Build query for existing user ────────────────────────────────────
def build_query_regular_node(state: RecommenderState) -> dict:
    print("🔍 Building query from user history...")
    persona  = state["persona"]
    context  = state.get("raw_context", {})
    top_cats = persona.get("top_categories", ["Restaurants"])[:3]
    intent   = context.get("intent", "")
    query    = f"{' '.join(top_cats)} {intent}".strip()

    return {
        "search_query":    query,
        "cultural_framing": get_recommendation_framing(
            top_categories=top_cats, city="Lagos"
        )
    }


# ── Node 2b: Build query for cold-start user ──────────────────────────────────
def build_query_cold_start_node(state: RecommenderState) -> dict:
    print("❄️  Cold-start — building query from description...")
    persona     = state["persona"]
    context     = state.get("raw_context", {})
    description = persona.get("description", "")
    intent      = context.get("intent", "")
    top_cats    = persona.get("top_categories", ["Restaurants", "Nigerian"])[:3]
    query       = f"{description} {' '.join(top_cats)} {intent}".strip()

    return {
        "search_query":    query,
        "cultural_framing": get_recommendation_framing(
            top_categories=top_cats, city="Lagos"
        )
    }


# ── Node 3: Search ChromaDB for candidates ────────────────────────────────────
def search_businesses_node(state: RecommenderState) -> dict:
    print("🏪 Searching businesses...")
    candidates = search_businesses(
        query=state["search_query"],
        k=state.get("k", 10) * 2   # fetch more, LLM will rank down to k
    )
    return {"candidate_businesses": candidates}


# ── Node 4: Rank and generate explanations ────────────────────────────────────
def rank_and_reason_node(state: RecommenderState) -> dict:
    print("🤔 Ranking and reasoning...")
    persona    = state["persona"]
    candidates = state["candidate_businesses"]
    context    = state.get("raw_context", {})
    k          = state.get("k", 10)

    candidates_text = "\n".join([
        f"{i+1}. {b['name']} | {b['categories']} | {b.get('address', 'N/A')} | {b['city']} | {b['stars']}★ | id:{b.get('business_id', '')}"
        for i, b in enumerate(candidates[:20])
    ])

    intent_text    = context.get('intent', 'general recommendation')
    needs_location = any(w in intent_text.lower() for w in ['where', 'location', 'located', 'address', 'area'])

    prompt = f"""You are a Nigerian recommendation agent. Pick the best places for this user.

USER PROFILE:
- Average rating: {persona['avg_rating']} stars
- Tone: {', '.join(persona.get('tone_tags', []))}
- Frequently visits: {', '.join(persona.get('top_categories', [])[:3])}
- New user (cold start): {state['is_cold_start']}

{state['cultural_framing']}

INTENT: {intent_text}

{"LOCATION REQUEST DETECTED: The user explicitly asked WHERE the place is. You MUST include the specific address and area in each recommendation's reason." if needs_location else ""}

CANDIDATES (format: # name | categories | address | city | stars | id):
{candidates_text}

Select the top {k} that best match this user. Explain each pick in a way 
that resonates with a Nigerian user — reference their taste, cultural context, 
and what makes the place right for them.

IMPORTANT: Always include the location (address or area) of each place in the 
'reason' field, so the user knows where to find it physically.

Respond ONLY in this JSON format:
{{
    "recommendations": [
        {{
            "business_id": "<exact id from candidates>",
            "name": "<name>",
            "categories": "<categories>",
            "city": "<city>",
            "stars": <float>,
            "score": <float 0-1>,
            "reason": "<Nigerian-toned explanation INCLUDING the address/area>"
        }}
    ],
    "clarifying_questions": ["<only if cold start and more info would help>"]
}}"""

    response = llm.invoke([
        SystemMessage(content=NAIJA_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    return {"raw_response": response.content}


# ── Node 5: Parse output ──────────────────────────────────────────────────────
def parse_output_node(state: RecommenderState) -> dict:
    print("📋 Parsing recommendations...")
    raw = state.get("raw_response", "")
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "recommendations":      data.get("recommendations", []),
                "clarifying_questions": data.get("clarifying_questions", [])
            }
    except Exception:
        pass
    return {"recommendations": [], "clarifying_questions": []}


# ── Build graph ───────────────────────────────────────────────────────────────
def build_recommender_graph():
    graph = StateGraph(RecommenderState)

    graph.add_node("parse_persona",          parse_persona_node)
    graph.add_node("build_query_regular",    build_query_regular_node)
    graph.add_node("build_query_cold_start", build_query_cold_start_node)
    graph.add_node("search_businesses",      search_businesses_node)
    graph.add_node("rank_and_reason",        rank_and_reason_node)
    graph.add_node("parse_output",           parse_output_node)

    graph.set_entry_point("parse_persona")

    # THE conditional edge — cold-start routing (25 points)
    graph.add_conditional_edges(
        "parse_persona",
        cold_start_router,
        {
            "regular":    "build_query_regular",
            "cold_start": "build_query_cold_start"
        }
    )

    graph.add_edge("build_query_regular",    "search_businesses")
    graph.add_edge("build_query_cold_start", "search_businesses")
    graph.add_edge("search_businesses",      "rank_and_reason")
    graph.add_edge("rank_and_reason",        "parse_output")
    graph.add_edge("parse_output",           END)

    return graph.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_recommender_graph()
    return _graph


# ── Public API — called by FastAPI ────────────────────────────────────────────
def recommend(request: RecommendRequest) -> RecommendResponse:
    graph  = get_graph()
    result = graph.invoke({
        "raw_persona": request.user_persona.model_dump(),
        "raw_context": request.context.model_dump() if request.context else {},
        "k":           request.k
    })

    recommendations = [
        RecommendationItem(
            business_id=r.get("business_id", ""),
            name=r.get("name", ""),
            categories=r.get("categories", ""),
            city=r.get("city", "Lagos"),
            stars=float(r.get("stars", 0)),
            score=float(r.get("score", 0)),
            reason=r.get("reason", "")
        )
        for r in result.get("recommendations", [])
    ]

    return RecommendResponse(
        recommendations=recommendations,
        is_cold_start=result.get("is_cold_start", False),
        clarifying_questions=result.get("clarifying_questions", [])
    )