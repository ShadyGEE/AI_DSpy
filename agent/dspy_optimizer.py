"""
DSPy optimization for NL→SQL module using BootstrapFewShot
"""
import dspy
from typing import List, Tuple
from agent.dspy_signatures import NLToSQLModule
from agent.tools import SQLiteTool


class SQLValidationMetric:
    """Metric to evaluate if generated SQL is valid and executable"""

    def __init__(self, db_tool: SQLiteTool):
        self.db_tool = db_tool

    def __call__(self, example, prediction, trace=None) -> float:
        """
        Evaluate the quality of generated SQL

        Returns:
            1.0 if SQL is valid and executes
            0.5 if SQL is syntactically valid but doesn't execute
            0.0 if SQL is invalid
        """
        try:
            sql = prediction.sql_query if hasattr(prediction, 'sql_query') else str(prediction)

            # Clean SQL
            sql = sql.strip()
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()

            # Validate syntax
            is_valid, error = self.db_tool.validate_query(sql)

            if not is_valid:
                return 0.0

            # Try to execute
            result = self.db_tool.execute_query(sql)

            if result["success"]:
                # Check if we got data
                if result["data"]:
                    return 1.0
                else:
                    return 0.8  # Valid but no results
            else:
                return 0.5  # Valid syntax but execution failed

        except Exception as e:
            return 0.0


def create_training_examples(schema: str) -> List[dspy.Example]:
    """
    Create training examples for NL→SQL optimization

    These are hand-crafted examples that represent common query patterns
    in the Northwind database
    """
    examples = [
        dspy.Example(
            question="What are the top 3 products by revenue?",
            schema=schema,
            constraints="{}",
            sql_query="""SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
FROM Products p
JOIN "Order Details" od ON p.ProductID = od.ProductID
GROUP BY p.ProductName
ORDER BY Revenue DESC
LIMIT 3"""
        ).with_inputs("question", "schema", "constraints"),

        dspy.Example(
            question="How many orders were placed in June 1997?",
            schema=schema,
            constraints='{"date_range": {"start": "1997-06-01", "end": "1997-06-30"}}',
            sql_query="""SELECT COUNT(DISTINCT OrderID) AS OrderCount
FROM Orders
WHERE OrderDate BETWEEN '1997-06-01' AND '1997-06-30'"""
        ).with_inputs("question", "schema", "constraints"),

        dspy.Example(
            question="What was the total revenue from Beverages category?",
            schema=schema,
            constraints='{"categories": ["Beverages"]}',
            sql_query="""SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
FROM "Order Details" od
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE c.CategoryName = 'Beverages'"""
        ).with_inputs("question", "schema", "constraints"),

        dspy.Example(
            question="Which category had the highest quantity sold?",
            schema=schema,
            constraints="{}",
            sql_query="""SELECT c.CategoryName, SUM(od.Quantity) AS TotalQuantity
FROM "Order Details" od
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
GROUP BY c.CategoryName
ORDER BY TotalQuantity DESC
LIMIT 1"""
        ).with_inputs("question", "schema", "constraints"),

        dspy.Example(
            question="What is the average order value in December 1997?",
            schema=schema,
            constraints='{"date_range": {"start": "1997-12-01", "end": "1997-12-31"}, "kpi_formula": "AOV"}',
            sql_query="""SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID) AS AOV
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
WHERE o.OrderDate BETWEEN '1997-12-01' AND '1997-12-31'"""
        ).with_inputs("question", "schema", "constraints"),

        dspy.Example(
            question="Who is the top customer by total revenue in 1997?",
            schema=schema,
            constraints='{"date_range": {"start": "1997-01-01", "end": "1997-12-31"}}',
            sql_query="""SELECT c.CompanyName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
FROM Customers c
JOIN Orders o ON c.CustomerID = o.CustomerID
JOIN "Order Details" od ON o.OrderID = od.OrderID
WHERE o.OrderDate BETWEEN '1997-01-01' AND '1997-12-31'
GROUP BY c.CompanyName
ORDER BY Revenue DESC
LIMIT 1"""
        ).with_inputs("question", "schema", "constraints"),

        # Additional examples for better coverage
        dspy.Example(
            question="Total revenue from Beverages in June 1997",
            schema=schema,
            constraints='{"date_range": {"start": "1997-06-01", "end": "1997-06-30"}, "categories": ["Beverages"]}',
            sql_query="""SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS Revenue
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE o.OrderDate BETWEEN '1997-06-01' AND '1997-06-30'
AND c.CategoryName = 'Beverages'"""
        ).with_inputs("question", "schema", "constraints"),

        dspy.Example(
            question="List all products in the Beverages category",
            schema=schema,
            constraints='{"categories": ["Beverages"]}',
            sql_query="""SELECT p.ProductName
FROM Products p
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE c.CategoryName = 'Beverages'"""
        ).with_inputs("question", "schema", "constraints"),
    ]

    return examples


def optimize_nl_to_sql(
    db_tool: SQLiteTool,
    lm: dspy.LM,
    num_examples: int = 8
) -> Tuple[NLToSQLModule, NLToSQLModule, dict]:
    """
    Optimize the NL→SQL module using BootstrapFewShot

    Returns:
        (unoptimized_module, optimized_module, metrics_dict)
    """
    # Set up DSPy
    dspy.configure(lm=lm)

    # Get schema
    schema = db_tool.get_schema()

    # Create training examples
    trainset = create_training_examples(schema)[:num_examples]

    # Create metric
    metric = SQLValidationMetric(db_tool)

    # Create modules
    unoptimized = NLToSQLModule()
    optimized = NLToSQLModule()

    # Measure baseline performance
    print("\n=== Evaluating Baseline (Unoptimized) ===")
    baseline_scores = []
    for example in trainset:
        try:
            pred = unoptimized(
                question=example.question,
                schema=example.schema,
                constraints=example.constraints
            )
            score = metric(example, type('Pred', (), {'sql_query': pred})())
            baseline_scores.append(score)
            print(f"Example: {example.question[:50]}... Score: {score}")
        except Exception as e:
            print(f"Error: {e}")
            baseline_scores.append(0.0)

    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    print(f"\nBaseline Average Score: {baseline_avg:.2f}")

    # Optimize using BootstrapFewShot
    print("\n=== Optimizing with BootstrapFewShot ===")
    try:
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )

        optimized = optimizer.compile(
            student=optimized,
            trainset=trainset
        )
        print("Optimization completed successfully!")

    except Exception as e:
        print(f"Optimization warning: {e}")
        print("Using unoptimized module as fallback")
        optimized = unoptimized

    # Measure optimized performance
    print("\n=== Evaluating Optimized Module ===")
    optimized_scores = []
    for example in trainset:
        try:
            pred = optimized(
                question=example.question,
                schema=example.schema,
                constraints=example.constraints
            )
            score = metric(example, type('Pred', (), {'sql_query': pred})())
            optimized_scores.append(score)
            print(f"Example: {example.question[:50]}... Score: {score}")
        except Exception as e:
            print(f"Error: {e}")
            optimized_scores.append(0.0)

    optimized_avg = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0
    print(f"\nOptimized Average Score: {optimized_avg:.2f}")

    # Calculate improvement
    improvement = optimized_avg - baseline_avg
    print(f"\n=== Optimization Results ===")
    print(f"Baseline:  {baseline_avg:.2f}")
    print(f"Optimized: {optimized_avg:.2f}")
    print(f"Improvement: {improvement:+.2f} ({(improvement/baseline_avg*100) if baseline_avg > 0 else 0:.1f}%)")

    metrics = {
        "baseline_score": baseline_avg,
        "optimized_score": optimized_avg,
        "improvement": improvement,
        "num_examples": len(trainset)
    }

    return unoptimized, optimized, metrics
