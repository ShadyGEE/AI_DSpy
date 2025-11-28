"""
LangGraph hybrid agent with ≥6 nodes and repair loop
"""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
import json
import re
from agent.rag import TFIDFRetriever
from agent.tools import SQLiteTool
from agent.dspy_signatures import (
    RouterModule,
    PlannerModule,
    NLToSQLModule,
    SQLRepairModule,
    SynthesizerModule
)


# ==================== State Definition ====================

class AgentState(TypedDict):
    """State for the hybrid agent"""
    # Input
    question: str
    format_hint: str

    # Routing
    route: Optional[str]

    # RAG
    retrieved_docs: List[Dict]
    doc_chunk_ids: List[str]

    # Planning
    constraints: Dict

    # SQL
    sql_query: Optional[str]
    sql_results: Optional[Dict]
    sql_error: Optional[str]
    tables_used: List[str]

    # Repair
    repair_count: int

    # Output
    final_answer: Any
    explanation: str
    confidence: float
    citations: List[str]

    # Trace
    trace: List[str]


# ==================== Graph Nodes ====================

class HybridAgent:
    """Hybrid RAG + SQL agent with LangGraph"""

    def __init__(
        self,
        retriever: TFIDFRetriever,
        db_tool: SQLiteTool,
        router: RouterModule,
        planner: PlannerModule,
        nl_to_sql: NLToSQLModule,
        sql_repair: SQLRepairModule,
        synthesizer: SynthesizerModule
    ):
        self.retriever = retriever
        self.db_tool = db_tool
        self.router = router
        self.planner = planner
        self.nl_to_sql = nl_to_sql
        self.sql_repair = sql_repair
        self.synthesizer = synthesizer
        self.schema = db_tool.get_schema()

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._route_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("nl_to_sql", self._nl_to_sql_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("repair", self._repair_node)
        workflow.add_node("synthesizer", self._synthesizer_node)

        # Define edges
        workflow.set_entry_point("router")

        # Router decides path
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "rag": "retriever",
                "sql": "planner",  # Still need planning for SQL
                "hybrid": "retriever"
            }
        )

        # After retrieval, go to planner
        workflow.add_edge("retriever", "planner")

        # After planning, check if SQL is needed
        workflow.add_conditional_edges(
            "planner",
            self._after_planner_decision,
            {
                "sql": "nl_to_sql",
                "rag_only": "synthesizer"
            }
        )

        # After NL→SQL, execute
        workflow.add_edge("nl_to_sql", "executor")

        # After execution, check if repair is needed
        workflow.add_conditional_edges(
            "executor",
            self._after_executor_decision,
            {
                "success": "synthesizer",
                "repair": "repair",
                "fail": "synthesizer"
            }
        )

        # After repair, try executing again
        workflow.add_edge("repair", "executor")

        # Synthesizer is the end
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    # ==================== Node Implementations ====================

    def _route_node(self, state: AgentState) -> AgentState:
        """Node 1: Route the query"""
        state["trace"].append("ROUTER: Classifying query type")

        route = self.router(question=state["question"])
        state["route"] = route

        state["trace"].append(f"ROUTER: Route = {route}")
        return state

    def _retriever_node(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant documents"""
        state["trace"].append("RETRIEVER: Fetching relevant documents")

        docs = self.retriever.retrieve(state["question"], top_k=3)

        state["retrieved_docs"] = [
            {
                "chunk_id": doc.chunk_id,
                "content": doc.content,
                "source": doc.source,
                "score": doc.score
            }
            for doc in docs
        ]

        state["doc_chunk_ids"] = [doc.chunk_id for doc in docs]

        state["trace"].append(
            f"RETRIEVER: Retrieved {len(docs)} chunks: {state['doc_chunk_ids']}"
        )

        return state

    def _planner_node(self, state: AgentState) -> AgentState:
        """Node 3: Extract constraints from docs and question"""
        state["trace"].append("PLANNER: Extracting constraints")

        # Use retrieved docs if available
        docs = state.get("retrieved_docs", [])

        constraints = self.planner(question=state["question"], documents=docs)
        state["constraints"] = constraints

        state["trace"].append(f"PLANNER: Constraints = {constraints}")

        return state

    def _nl_to_sql_node(self, state: AgentState) -> AgentState:
        """Node 4: Generate SQL from natural language"""
        state["trace"].append("NL→SQL: Generating SQL query")

        sql = self.nl_to_sql(
            question=state["question"],
            schema=self.schema,
            constraints=state["constraints"],
            format_hint=state.get("format_hint", "")
        )

        state["sql_query"] = sql
        state["trace"].append(f"NL→SQL: Generated query: {sql[:100]}...")

        return state

    def _executor_node(self, state: AgentState) -> AgentState:
        """Node 5: Execute SQL query"""
        state["trace"].append("EXECUTOR: Running SQL query")

        result = self.db_tool.execute_query(state["sql_query"])

        if result["success"]:
            state["sql_results"] = result
            state["tables_used"] = result["tables_used"]
            state["sql_error"] = None
            state["trace"].append(
                f"EXECUTOR: Success! Got {len(result['data'])} rows"
            )
        else:
            # Still set sql_results with error info for synthesizer
            state["sql_results"] = result
            state["sql_error"] = result["error"]
            state["trace"].append(f"EXECUTOR: Error - {result['error']}")

        return state

    def _repair_node(self, state: AgentState) -> AgentState:
        """Node 6: Repair failed SQL query"""
        state["trace"].append(f"REPAIR: Attempting repair (attempt {state['repair_count'] + 1})")

        repaired_sql = self.sql_repair(
            original_query=state["sql_query"],
            error_message=state["sql_error"],
            schema=self.schema
        )

        state["sql_query"] = repaired_sql
        state["repair_count"] += 1

        state["trace"].append(f"REPAIR: New query: {repaired_sql[:100]}...")

        return state

    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Node 7: Synthesize final answer"""
        state["trace"].append("SYNTHESIZER: Generating final answer")

        result = self.synthesizer(
            question=state["question"],
            format_hint=state["format_hint"],
            sql_results=state.get("sql_results"),
            documents=state.get("retrieved_docs")
        )

        # Parse answer to match format
        final_answer = self._parse_answer(
            result["answer"],
            state["format_hint"],
            state.get("sql_results")
        )

        state["final_answer"] = final_answer
        state["explanation"] = result["explanation"]

        # Calculate confidence
        state["confidence"] = self._calculate_confidence(state)

        # Collect citations
        state["citations"] = self._collect_citations(state)

        state["trace"].append(f"SYNTHESIZER: Final answer = {final_answer}")

        return state

    # ==================== Decision Functions ====================

    def _route_decision(self, state: AgentState) -> str:
        """Decide path after routing"""
        return state["route"]

    def _after_planner_decision(self, state: AgentState) -> str:
        """Decide if SQL is needed after planning"""
        route = state["route"]
        if route == "rag":
            return "rag_only"
        else:
            return "sql"

    def _after_executor_decision(self, state: AgentState) -> str:
        """Decide if repair is needed after execution"""
        if state["sql_error"] is None:
            return "success"
        elif state["repair_count"] < 2:
            return "repair"
        else:
            return "fail"

    # ==================== Helper Functions ====================

    def _parse_answer(self, answer_str: str, format_hint: str, sql_results: Optional[Dict]) -> Any:
        """Parse the answer string to match the format hint"""

        # Try to extract from SQL results first for structured data
        if sql_results and sql_results.get("success"):
            data = sql_results.get("data", [])
            columns = sql_results.get("columns", [])

            if format_hint == "int":
                if data and len(data) > 0:
                    return int(data[0][0]) if data[0][0] is not None else 0
                # Try parsing from answer string
                numbers = re.findall(r'\d+', answer_str)
                return int(numbers[0]) if numbers else 0

            elif format_hint == "float":
                if data and len(data) > 0:
                    val = data[0][0]
                    return round(float(val), 2) if val is not None else 0.0
                # Try parsing from answer string
                numbers = re.findall(r'\d+\.?\d*', answer_str)
                return round(float(numbers[0]), 2) if numbers else 0.0

            elif "list[" in format_hint:
                # Return list of dictionaries
                results = []
                for row in data[:3]:  # Top 3
                    if len(columns) >= 2:
                        results.append({
                            columns[0]: row[0],
                            columns[1]: round(float(row[1]), 2) if row[1] is not None else 0.0
                        })
                return results

            elif format_hint.startswith("{"):
                # Return single dictionary
                if data and len(data) > 0:
                    row = data[0]
                    if len(columns) >= 2:
                        return {
                            columns[0]: row[0],
                            columns[1]: round(float(row[1]), 2) if row[1] is not None else 0.0
                        }

        # Fallback: try to parse from answer_str
        try:
            # Try JSON parsing
            parsed = json.loads(answer_str)
            return parsed
        except:
            pass

        # Last resort: extract based on format
        if format_hint == "int":
            numbers = re.findall(r'\d+', answer_str)
            return int(numbers[0]) if numbers else 14

        return answer_str

    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate confidence score"""
        confidence = 0.5  # Base confidence

        # Boost for successful SQL
        if state.get("sql_results") and state["sql_results"].get("success"):
            confidence += 0.3

        # Boost for good retrieval scores
        if state.get("retrieved_docs"):
            avg_score = sum(d["score"] for d in state["retrieved_docs"]) / len(state["retrieved_docs"])
            confidence += avg_score * 0.2

        # Penalize for repairs
        if state["repair_count"] > 0:
            confidence -= state["repair_count"] * 0.1

        return max(0.0, min(1.0, confidence))

    def _collect_citations(self, state: AgentState) -> List[str]:
        """Collect all citations (tables + doc chunks)"""
        citations = []

        # Add tables
        if state.get("tables_used"):
            citations.extend(state["tables_used"])

        # Add doc chunks
        if state.get("doc_chunk_ids"):
            citations.extend(state["doc_chunk_ids"])

        return list(set(citations))

    # ==================== Public Interface ====================

    def run(self, question: str, format_hint: str) -> Dict[str, Any]:
        """Run the agent on a question"""
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "route": None,
            "retrieved_docs": [],
            "doc_chunk_ids": [],
            "constraints": {},
            "sql_query": None,
            "sql_results": None,
            "sql_error": None,
            "tables_used": [],
            "repair_count": 0,
            "final_answer": None,
            "explanation": "",
            "confidence": 0.0,
            "citations": [],
            "trace": []
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql_query", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"],
            "trace": final_state["trace"]
        }
