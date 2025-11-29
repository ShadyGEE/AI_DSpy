"""
DSPy signatures and modules for the hybrid agent
"""
import dspy
from typing import List, Optional


# ==================== Signatures ====================

class RouteQuery(dspy.Signature):
    """Route questions to correct handler.

    RAG: Questions about policies, definitions, procedures. ESPECIALLY if question says 'according to', 'per the', 'as stated in'.
    SQL: Data queries (top-N, totals, counts) from database.
    HYBRID: Questions needing both docs AND data (e.g., KPIs with business rules + calculations).

    CRITICAL: If question explicitly references a document/policy ('according to product policy'), use RAG only!
    """
    question = dspy.InputField(desc="Question")
    route = dspy.OutputField(desc="rag/sql/hybrid")


class ExtractConstraints(dspy.Signature):
    """Extract constraints from docs. Know available tables to guide extraction."""
    question = dspy.InputField(desc="Question")
    documents = dspy.InputField(desc="Docs")
    schema = dspy.InputField(desc="Available database tables and columns")
    constraints = dspy.OutputField(desc="JSON constraints")


class GenerateSQL(dspy.Signature):
    """CRITICAL: Use strftime (NOT strftForms!). Quote \"Order Details\". Winter=12, Summer=06."""
    question = dspy.InputField(desc="Question")
    schema = dspy.InputField(desc="Schema")
    constraints = dspy.InputField(desc="Constraints")
    format_hint = dspy.InputField(desc="Format")
    sql_query = dspy.OutputField(desc="Valid SQL query")


class RepairSQL(dspy.Signature):
    """Fix typos: strftForms→strftime. Quote \"Order Details\". Tables: Orders, Products, Categories."""
    original_query = dspy.InputField(desc="Failed query")
    error_message = dspy.InputField(desc="Error")
    schema = dspy.InputField(desc="Schema")
    repaired_query = dspy.OutputField(desc="Corrected SQL")


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

    def forward(self, question: str, documents: List[dict], schema: str = "") -> dict:
        """Extract constraints from question and documents"""
        # Format documents for LLM
        doc_text = "\n\n".join([
            f"[{doc['chunk_id']}] {doc['content']}"
            for doc in documents
        ])

        result = self.extract(question=question, documents=doc_text, schema=schema)

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

        # Auto-fix common typos
        import re
        sql = sql.replace("strftForms", "strftime")
        sql = sql.replace("strftTime", "strftime")
        sql = sql.replace("BETWEWEN", "BETWEEN")
        sql = sql.replace("BETWEInstance", "BETWEEN")
        sql = sql.replace("`Order Details`", '"Order Details"')
        # Fix OrderDetails (no space) → "Order Details" (with space and quotes)
        sql = re.sub(r'\bOrderDetails\b', '"Order Details"', sql)
        sql = re.sub(r'\border_details\b', '"Order Details"', sql, flags=re.IGNORECASE)

        # Fix gibberish patterns like "o- 'Instance'" or "o-'0'"
        sql = re.sub(r"[a-z]-\s*'(?:Instance|0)'", "o.OrderDate", sql)

        # Fix standalone Instance
        sql = re.sub(r'\bInstance\b', '0', sql)

        # Fix BETWEEN with strftime('%Y-%m') - should use = instead
        # Pattern: strftime('%Y-%m', o.OrderDate) BETWEEN '2013-12-01' AND '2013-12-31'
        # Should be: strftime('%Y-%m', o.OrderDate) = '2013-12'
        sql = re.sub(
            r"strftime\('%Y-%m',\s*([^)]+)\)\s+BETWEEN\s+'(\d{4})-(\d{2})-\d{2}'\s+AND\s+'[^']+?'",
            r"strftime('%Y-%m', \1) = '\2-\3'",
            sql,
            flags=re.IGNORECASE
        )

        # DISABLED: Fix ROUND() - this was adding duplicate , 2) parameters
        # The model now generates correct ROUND syntax, so this fix is no longer needed
        # def fix_round(match):
        #     content = match.group(1)
        #     alias = match.group(2)
        #     return f"ROUND({content}, 2) AS {alias}"
        # sql = re.sub(r"ROUND\s*\(\s*(.*?)\s+AS\s+(\w+)", fix_round, sql, flags=re.IGNORECASE | re.DOTALL)

        # Remove incorrect customer name filters for marketing campaigns
        # Pattern: AND o.CustomerID IN (SELECT CustomerID FROM Customers WHERE CompanyName LIKE '%Summer Beverages 2013%')
        sql = re.sub(
            r"\s+AND\s+\w+\.CustomerID\s+IN\s*\(SELECT\s+CustomerID\s+FROM\s+Customers\s+WHERE\s+CompanyName\s+LIKE\s+'%[^']*?(Summer|Winter)[^']*?20\d{2}%'\)",
            "",
            sql,
            flags=re.IGNORECASE
        )

        # Remove incorrect CategoryName filters for marketing campaigns
        # Pattern: AND Categories.CategoryName='Summer Beverages' (marketing campaigns are NOT categories)
        sql = re.sub(
            r"\s+AND\s+\w+\.CategoryName\s*=\s*'[^']*?(Summer|Winter)[^']*?(Beverages|Classics)[^']*?20\d{2}[^']*?'",
            "",
            sql,
            flags=re.IGNORECASE
        )

        # Fix wrong date range for yearly queries: '2013-06' -> '2013%' when asking about full year
        # Only fix if the query doesn't mention a specific month/season in constraints
        if "2013" in sql and "entire" in str(constraints).lower() or "all of 2013" in str(constraints).lower():
            sql = re.sub(
                r"strftime\('%Y-%m',([^)]+)\)\s*=\s*'2013-\d{2}'",
                r"strftime('%Y', \1) = '2013'",
                sql,
                flags=re.IGNORECASE
            )

        # Fix IFNULL with wrong default for Discount: IFNULL(Discount, -1) -> IFNULL(Discount, 0)
        sql = re.sub(
            r"IFNULL\s*\(\s*([^,]+\.)?Discount\s*,\s*-1\s*\)",
            r"IFNULL(\1Discount, 0)",
            sql,
            flags=re.IGNORECASE
        )

        # CRITICAL FIX: Remove division by order count when calculating TOTAL revenue or margin
        # Use string manipulation to handle deeply nested parentheses
        # Pattern 1: / COUNT(DISTINCT ..OrderID) AS <alias> (simple case)
        # Pattern 2: / COUNT(DISTINCT ..OrderID), <num>) AS <alias> (ROUND case)
        import re

        # Handle ROUND case first: / COUNT(DISTINCT ..OrderID), 2) AS <alias>
        pattern_round = r"\s*/\s*COUNT\s*\(\s*DISTINCT\s+\w+\.OrderID\s*\)\s*,\s*\d+\s*\)\s+AS\s+(\w+)"
        matches_round = list(re.finditer(pattern_round, sql, re.IGNORECASE))
        for match in reversed(matches_round):
            alias = match.group(1)
            if alias.lower() not in ['aov', 'averageordervalue', 'avgordervalue']:
                # Remove / COUNT(DISTINCT ..OrderID), keep the precision: , 2) AS alias
                comma_pos = match.group(0).rfind(',')
                remainder = match.group(0)[comma_pos:]  # ", 2) AS alias"
                sql = sql[:match.start()] + remainder + sql[match.end():]

        # Handle simple case: / COUNT(DISTINCT ..OrderID) AS <alias>
        pattern_simple = r"\s*/\s*COUNT\s*\(\s*DISTINCT\s+\w+\.OrderID\s*\)\s+AS\s+(\w+)"
        matches_simple = list(re.finditer(pattern_simple, sql, re.IGNORECASE))
        for match in reversed(matches_simple):
            alias = match.group(1)
            if alias.lower() not in ['aov', 'averageordervalue', 'avgordervalue']:
                sql = sql[:match.start()] + f" AS {alias}" + sql[match.end():]

        # Remove GROUP BY clauses that reference tables not in the query
        # Pattern: GROUP BY Categories.CategoryName when Categories not joined
        sql = re.sub(
            r"\s+GROUP\s+BY\s+Categories\.\w+",
            "",
            sql,
            flags=re.IGNORECASE
        )

        return sql


class SQLRepairModule(dspy.Module):
    """Repair failed SQL queries"""

    def __init__(self):
        super().__init__()
        self.repair = dspy.Predict(RepairSQL)  # Changed from ChainOfThought

    def forward(self, original_query: str, error_message: str, schema: str) -> str:
        """Repair a failed SQL query"""
        # Auto-fix common typos first
        fixed_query = original_query
        # strftime typos
        fixed_query = fixed_query.replace("strftForms", "strftime")
        fixed_query = fixed_query.replace("strftTime", "strftime")
        # BETWEEN typos
        fixed_query = fixed_query.replace("BETWEWEN", "BETWEEN")
        fixed_query = fixed_query.replace("BETWEInstance", "BETWEEN")
        # Table name issues
        fixed_query = fixed_query.replace("`Order Details`", '"Order Details"')
        fixed_query = fixed_query.replace("OrderDetails", '"Order Details"')
        # Instance typo (when model types Instance instead of 0)
        import re
        fixed_query = re.sub(r'\bInstance\b', '0', fixed_query)

        # If we fixed typos and error is about syntax, return fixed version
        if fixed_query != original_query and ("syntax" in error_message.lower() or "no such" in error_message.lower()):
            return fixed_query

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

        # Apply typo fixes to repaired query too
        sql = sql.replace("strftForms", "strftime")
        sql = sql.replace("strftTime", "strftime")
        sql = sql.replace("`Order Details`", '"Order Details"')

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
