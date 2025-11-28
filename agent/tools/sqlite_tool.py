"""
SQLite database tools for Northwind database
"""
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class SQLiteTool:
    """Tool for interacting with Northwind SQLite database"""

    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise ValueError(f"Database not found: {self.db_path}")
        self._schema_cache = None  # Always regenerate schema to pick up changes

    def get_connection(self):
        """Get a connection to the database"""
        return sqlite3.connect(self.db_path)

    def get_schema(self) -> str:
        """Get compact database schema with JOIN examples"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]

        schema_lines = [
            "Tables: Categories, Products, \"Order Details\" (MUST quote!), Orders, Customers",
            "Date: strftime('%Y-%m',Orders.OrderDate)='2013-06'. Revenue: SUM(UnitPrice*Quantity*(1-Discount))",
            "AOV: Revenue/COUNT(DISTINCT OrderID). CostOfGoods: 0.7*UnitPrice"
        ]

        # Only show key tables
        key_tables = ["Categories", "Products", "Order Details", "Orders", "Customers"]
        for table in tables:
            if table not in key_tables:
                continue

            cursor.execute(f"PRAGMA table_info([{table}]);")
            columns = cursor.fetchall()

            # Quote table name if it has spaces
            table_name = f'"{table}"' if ' ' in table else table

            # Only show 3 key columns
            cols = []
            for col in columns[:3]:
                col_id, name, col_type, not_null, default, pk = col
                cols.append(name)

            schema_lines.append(f"{table_name}({', '.join(cols)}...)")

            # Add JOIN hints for FKs
            cursor.execute(f"PRAGMA foreign_key_list([{table}]);")
            fks = cursor.fetchall()
            for fk in fks:
                _, _, ref_table, from_col, to_col, _, _, _ = fk
                if ref_table in key_tables:
                    ref_name = f'"{ref_table}"' if ' ' in ref_table else ref_table
                    schema_lines.append(f"  {from_col}->{ref_name}.{to_col}")

        conn.close()

        return "\n".join(schema_lines)

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> Dict[str, Any]:
        """
        Execute a SQL query and return results with metadata

        Returns:
            Dict with keys: success, data, columns, error, tables_used
        """
        result = {
            "success": False,
            "data": [],
            "columns": [],
            "error": None,
            "tables_used": []
        }

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Get column names
            if cursor.description:
                result["columns"] = [desc[0] for desc in cursor.description]

            # Get data
            result["data"] = cursor.fetchall()
            result["success"] = True

            # Extract tables used from query
            result["tables_used"] = self._extract_tables_from_query(query)

            conn.close()

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False

        return result

    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from SQL query"""
        query_upper = query.upper()
        tables = []

        # Common table names in Northwind
        known_tables = [
            "Orders", "Order Details", "Products", "Customers",
            "Employees", "Categories", "Suppliers", "Shippers",
            "orders", "order_items", "products", "customers"
        ]

        for table in known_tables:
            # Check if table name appears in query
            if table.upper() in query_upper:
                # Avoid duplicates
                canonical = table if table[0].isupper() else table.capitalize()
                if "Order Details" in query or "order_items" in query.lower():
                    if "Order Details" not in tables:
                        tables.append("Order Details")
                elif canonical not in tables and canonical != "Order Details":
                    tables.append(canonical)

        return list(set(tables))

    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a SQL query without executing it

        Returns:
            (is_valid, error_message)
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            # Use EXPLAIN to validate without executing
            cursor.execute(f"EXPLAIN {query}")
            conn.close()
            return True, None
        except Exception as e:
            return False, str(e)

    def get_table_sample(self, table_name: str, limit: int = 5) -> Dict[str, Any]:
        """Get sample rows from a table"""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)
