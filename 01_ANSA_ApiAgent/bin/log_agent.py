"""NL-to-SQL Log Analytics Agent.

Converts natural language questions about usage data into SQL queries
against the SQLite log database. Supports questions like:
- "How many requests today?"
- "Which model is used most?"
- "Show me errors from the last hour"
- "What are the top 5 user queries this week?"

Usage:
    from bin.log_agent import LogAgent
    
    agent = LogAgent()
    result = agent.query("How many requests were made today?")
"""

import json
import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import LLMClient, create_client, settings
from bin.admin_tools import run_sql, DB_SCHEMA


logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt for NL-to-SQL
# =============================================================================

NL_TO_SQL_PROMPT = f"""You are a SQL expert. Convert natural language questions into SQLite SELECT queries.

Database schema:
{DB_SCHEMA}

Rules:
1. ONLY generate SELECT queries. Never INSERT, UPDATE, DELETE, DROP.
2. Use strftime() for date/time operations (SQLite syntax).
3. Use 'now' for current time: datetime('now'), date('now'), strftime('%H', 'now')
4. Return ONLY the SQL query, no explanation. No markdown code blocks.
5. Limit results to 50 rows max unless the user asks for more.
6. Use aliases for readability (e.g., COUNT(*) as total).

Examples:
- "how many requests today?" -> SELECT COUNT(*) as total FROM requests WHERE date(timestamp) = date('now')
- "which model is most used?" -> SELECT model, COUNT(*) as count FROM requests GROUP BY model ORDER BY count DESC LIMIT 5
- "average latency this week" -> SELECT ROUND(AVG(latency_ms), 1) as avg_latency_ms FROM requests WHERE timestamp >= datetime('now', '-7 days')
- "show recent errors" -> SELECT timestamp, model, error_message FROM errors ORDER BY timestamp DESC LIMIT 10
- "tool success rates" -> SELECT tool_name, ROUND(100.0 * SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) / COUNT(*), 1) as success_pct FROM tool_calls GROUP BY tool_name
"""


# =============================================================================
# Log Analytics Agent
# =============================================================================

class LogAgent:
    """Natural language to SQL agent for querying usage logs."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the log agent.
        
        Args:
            llm_client: Pre-configured LLM client (creates one if None)
        """
        self.client = llm_client or create_client()
    
    def query(self, question: str) -> dict:
        """Convert a natural language question to SQL, execute it, and return results.
        
        Args:
            question: Natural language question about usage data
        
        Returns:
            Dict with {question, sql, results, row_count, error}
        """
        # Step 1: Generate SQL from NL
        sql = self._generate_sql(question)
        
        if not sql:
            return {
                "question": question,
                "sql": "",
                "results": [],
                "row_count": 0,
                "error": "Could not generate SQL query.",
            }
        
        # Step 2: Execute the SQL
        results = run_sql(sql)
        
        # Check for error
        if results and "error" in results[0]:
            return {
                "question": question,
                "sql": sql,
                "results": [],
                "row_count": 0,
                "error": results[0]["error"],
            }
        
        return {
            "question": question,
            "sql": sql,
            "results": results,
            "row_count": len(results),
            "error": None,
        }
    
    def _generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question."""
        messages = [
            {"role": "system", "content": NL_TO_SQL_PROMPT},
            {"role": "user", "content": question},
        ]
        
        try:
            response = self.client.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            
            sql = response.content.strip()
            
            # Clean up: remove markdown code blocks if present
            if sql.startswith("```"):
                lines = sql.split("\n")
                sql = "\n".join(lines[1:-1])
            
            sql = sql.strip().rstrip(";")
            
            # Validate it's a SELECT
            if not sql.upper().startswith("SELECT"):
                logger.warning(f"Generated non-SELECT query: {sql[:50]}")
                return ""
            
            logger.info(f"NL-to-SQL: '{question[:50]}' -> {sql[:80]}")
            return sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return ""
    
    def format_results(self, query_result: dict) -> str:
        """Format query results as a readable string.
        
        Args:
            query_result: Dict from self.query()
        
        Returns:
            Formatted string for display
        """
        if query_result["error"]:
            return f"Error: {query_result['error']}\nSQL: {query_result['sql']}"
        
        if not query_result["results"]:
            return f"No results.\nSQL: {query_result['sql']}"
        
        # Format as table
        results = query_result["results"]
        headers = list(results[0].keys())
        
        # Column widths
        widths = {h: len(h) for h in headers}
        for row in results:
            for h in headers:
                widths[h] = max(widths[h], len(str(row.get(h, ""))))
        
        # Build table
        lines = []
        header_line = " | ".join(h.ljust(widths[h]) for h in headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        for row in results[:50]:
            line = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
            lines.append(line)
        
        table = "\n".join(lines)
        return f"SQL: {query_result['sql']}\n\nResults ({query_result['row_count']} rows):\n\n{table}"


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = LogAgent()
    
    print("\nLog Analytics Agent (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if question.lower() in ("quit", "exit", "q"):
            break
        
        result = agent.query(question)
        print(agent.format_results(result))
