"""
Mock LM for testing without Ollama
Generates simple rule-based SQL queries
"""
import dspy


class MockLM(dspy.LM):
    """Mock language model for testing"""

    def __init__(self):
        super().__init__(model="mock")

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Generate mock responses based on keywords in prompt"""
        if messages:
            prompt = str(messages)
        elif not prompt:
            prompt = ""

        prompt_lower = prompt.lower()

        # Router
        if "classify" in prompt_lower or "route" in prompt_lower:
            if "return" in prompt_lower and "policy" in prompt_lower:
                return ['{"reasoning": "This is a RAG question about policies.", "route": "rag"}']
            elif "revenue" in prompt_lower or "top" in prompt_lower:
                return ['{"reasoning": "This needs both SQL and RAG.", "route": "hybrid"}']
            else:
                return ['{"reasoning": "Using hybrid approach.", "route": "hybrid"}']

        # NL to SQL
        if "generate" in prompt_lower and "sql" in prompt_lower:
            if "top 3 products" in prompt_lower and "revenue" in prompt_lower:
                sql = """SELECT p.ProductName, ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS Revenue
FROM Products p
JOIN "Order Details" od ON p.ProductID = od.ProductID
GROUP BY p.ProductName
ORDER BY Revenue DESC
LIMIT 3"""
                return [sql]

            elif "category" in prompt_lower and "quantity" in prompt_lower and "2013-06" in prompt_lower:
                sql = """SELECT c.CategoryName, SUM(od.Quantity) AS TotalQuantity
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE o.OrderDate BETWEEN '2013-06-01' AND '2013-06-30'
GROUP BY c.CategoryName
ORDER BY TotalQuantity DESC
LIMIT 1"""
                return [sql]

            elif "aov" in prompt_lower or "average order value" in prompt_lower:
                if "2013-12" in prompt_lower:
                    sql = """SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) AS AOV
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
WHERE o.OrderDate BETWEEN '2013-12-01' AND '2013-12-31'"""
                    return [sql]

            elif "beverages" in prompt_lower and "revenue" in prompt_lower:
                if "2013-06" in prompt_lower:
                    sql = """SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS Revenue
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE o.OrderDate BETWEEN '2013-06-01' AND '2013-06-30'
AND c.CategoryName = 'Beverages'"""
                    return [sql]

            elif "customer" in prompt_lower and "margin" in prompt_lower:
                if "2013" in prompt_lower:
                    sql = """SELECT c.CompanyName, ROUND(SUM((od.UnitPrice - 0.7 * od.UnitPrice) * od.Quantity * (1 - od.Discount)), 2) AS Margin
FROM Customers c
JOIN Orders o ON c.CustomerID = o.CustomerID
JOIN "Order Details" od ON o.OrderID = od.OrderID
WHERE strftime('%Y', o.OrderDate) = '2013'
GROUP BY c.CompanyName
ORDER BY Margin DESC
LIMIT 1"""
                    return [sql]

            # Generic fallback
            return ["SELECT * FROM Orders LIMIT 1"]

        # Synthesizer
        if "synthesize" in prompt_lower or "final answer" in prompt_lower:
            if "int" in prompt_lower:
                return ["14"]
            elif "float" in prompt_lower:
                return ["1234.56"]
            else:
                return ["Answer based on the data"]

        # Planner/constraints
        if "extract" in prompt_lower and "constraints" in prompt_lower:
            if "2013-06" in prompt_lower or "summer" in prompt_lower:
                return ['{"date_range": {"start": "2013-06-01", "end": "2013-06-30"}, "categories": ["Beverages"]}']
            elif "2013-12" in prompt_lower or "winter" in prompt_lower:
                return ['{"date_range": {"start": "2013-12-01", "end": "2013-12-31"}}']
            else:
                return ['{}']

        # Default
        return ["OK"]

    def inspect_history(self, n=1):
        return []
