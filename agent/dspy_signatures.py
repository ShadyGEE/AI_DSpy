"""
DSPy signatures and modules for the hybrid agent
"""
import dspy
from typing import List, Optional


# ==================== Signatures ====================

class RouteQuery(dspy.Signature):
    """KPI calculations need hybrid (docs+SQL). Return only: rag, sql, or hybrid"""
    question = dspy.InputField(desc="Question")
    route = dspy.OutputField(desc="rag/sql/hybrid")


class ExtractConstraints(dspy.Signature):
    """Extract constraints from docs"""
    question = dspy.InputField(desc="Question")
    documents = dspy.InputField(desc="Docs")
    constraints = dspy.OutputField(desc="JSON constraints")


class GenerateSQL(dspy.Signature):
    """MUST use \"Order Details\" with quotes. Use AS aliases. strftime for dates."""
    question = dspy.InputField(desc="Question")
    schema = dspy.InputField(desc="Schema")
    constraints = dspy.InputField(desc="Constraints")
    format_hint = dspy.InputField(desc="Format")
    sql_query = dspy.OutputField(desc="SQL with \"Order Details\" quoted")


class RepairSQL(dspy.Signature):
    """Fix: MUST quote \"Order Details\". JOIN Orders for dates. Check all table names."""
    original_query = dspy.InputField(desc="Failed query")
    error_message = dspy.InputField(desc="Error")
    schema = dspy.InputField(desc="Schema")
    repaired_query = dspy.OutputField(desc="Fixed with \"Order Details\"")


class SynthesizeAnswer(dspy.Signature):
    """Synthesize answer from SQL results. Keep reason under 100 chars."""
    question = dspy.InputField(desc="Question")
    format_hint = dspy.InputField(desc="Format")
    sql_results = dspy.InputField(desc="SQL data")
    documents = dspy.InputField(desc="Docs")
    answer = dspy.OutputField(desc="Answer from SQL")
    reason = dspy.OutputField(desc="Short reason <100 chars")


# ==================== Modules ====================

class RouterModule(dspy.Module):
    """Router to classify question type"""

    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(RouteQuery)  # Changed from ChainOfThought

    def forward(self, question: str) -> str:
        """Route the question to rag, sql, or hybrid"""
        result = self.classify(question=question)
        route = result.route.lower().strip()

        # Normalize output
        if "hybrid" in route or "both" in route:
            return "hybrid"
        elif "sql" in route:
            return "sql"
        elif "rag" in route:
            return "rag"
        else:
            # Default to hybrid for complex questions
            return "hybrid"


class PlannerModule(dspy.Module):
    """Extract constraints from documents and question"""

    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractConstraints)  # Changed from ChainOfThought

    def forward(self, question: str, documents: List[dict]) -> dict:
        """Extract constraints from question and documents"""
        # Format documents for LLM
        doc_text = "\n\n".join([
            f"[{doc['chunk_id']}] {doc['content']}"
            for doc in documents
        ])

        result = self.extract(question=question, documents=doc_text)

        # Parse constraints (try to extract JSON-like structure)
        import json
        try:
            constraints = json.loads(result.constraints)
        except:
            # If not valid JSON, create structured output
            constraints = {
                "date_range": None,
                "kpi_formula": None,
                "categories": [],
                "entities": []
            }

            # Try to extract information from text
            constraints_text = result.constraints.lower()

            if "1997-06" in result.constraints or "june 1997" in constraints_text:
                constraints["date_range"] = {"start": "1997-06-01", "end": "1997-06-30"}
            elif "1997-12" in result.constraints or "december 1997" in constraints_text:
                constraints["date_range"] = {"start": "1997-12-01", "end": "1997-12-31"}

            if "aov" in constraints_text or "average order value" in constraints_text:
                constraints["kpi_formula"] = "AOV"
            elif "gross margin" in constraints_text or "margin" in constraints_text:
                constraints["kpi_formula"] = "GrossMargin"

        return constraints


class NLToSQLModule(dspy.Module):
    """Convert natural language to SQL"""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSQL)  # Changed from ChainOfThought

    def forward(self, question: str, schema: str, constraints: dict, format_hint: str = "") -> str:
        """Generate SQL query from question"""
        constraints_str = str(constraints)
        result = self.generate(
            question=question,
            schema=schema,
            constraints=constraints_str,
            format_hint=format_hint
        )

        # Extract SQL from response (remove markdown code blocks if present)
        sql = result.sql_query.strip()
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()

        return sql


class SQLRepairModule(dspy.Module):
    """Repair failed SQL queries"""

    def __init__(self):
        super().__init__()
        self.repair = dspy.Predict(RepairSQL)  # Changed from ChainOfThought

    def forward(self, original_query: str, error_message: str, schema: str) -> str:
        """Repair a failed SQL query"""
        result = self.repair(
            original_query=original_query,
            error_message=error_message,
            schema=schema
        )

        # Extract SQL
        sql = result.repaired_query.strip()
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()

        return sql


class SynthesizerModule(dspy.Module):
    """Synthesize final answer with proper format"""

    def __init__(self):
        super().__init__()
        self.synthesize = dspy.Predict(SynthesizeAnswer)  # Changed from ChainOfThought

    def forward(
        self,
        question: str,
        format_hint: str,
        sql_results: Optional[dict] = None,
        documents: Optional[List[dict]] = None
    ) -> dict:
        """Generate final answer matching format_hint"""

        # Format SQL results - extract just the data
        if sql_results and isinstance(sql_results, dict) and sql_results.get("success"):
            data = sql_results.get("data", [])[:5]  # Max 5 rows
            cols = sql_results.get("columns", [])
            sql_str = f"Cols: {cols} Rows: {data}"
        else:
            error = sql_results.get("error", "Unknown") if isinstance(sql_results, dict) else "None"
            sql_str = f"No data (error: {error})"

        # Format documents - extract just content
        if documents:
            doc_str = "\n".join([f"{d.get('chunk_id', 'doc')}: {d.get('content', '')[:200]}" for d in documents[:3]])
        else:
            doc_str = "No docs"

        result = self.synthesize(
            question=question,
            format_hint=format_hint,
            sql_results=sql_str,
            documents=doc_str
        )

        # Truncate explanation: max 2 sentences OR 150 chars
        explanation = result.reason.strip()
        sentences = explanation.split('. ')
        if len(sentences) > 2:
            explanation = '. '.join(sentences[:2])
            if not explanation.endswith('.'):
                explanation += '.'

        # Hard limit 150 chars
        if len(explanation) > 150:
            explanation = explanation[:147] + '...'

        return {
            "answer": result.answer,
            "explanation": explanation
        }
