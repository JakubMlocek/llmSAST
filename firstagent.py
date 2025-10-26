# agents.py
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# Prompts (bez zmian merytorycznych)
# =========================

SYSTEM_PROMPT = """
You are a senior software architect acting as a "Code Context Analyzer".
Your task is to analyze source code and generate a concise summary of its
business and technical context.

You will receive the content of a file. Your goal is to answer the following questions
in a coherent, 2-3 paragraph summary in English:

1.  What is the main purpose or responsibility of this code? (e.g., "Handling user login",
    "Product data model", "Interface to a payment API").
2.  What key input data or data types does it process?
    (e.g., "Login credentials (email, password)", "Session ID", "Credit card data").
3.  What other external components (of the system or the world) does it seem to
    communicate with? (e.g., "Database (sessions, users)", "External authorization API",
    "File system").

Your analysis must be concise and focused solely on the context,
ignoring implementation details that do not affect the understanding of its role.
"""

SYSTEM_PROMPT_DIFF = """
You are "Parser Zmian" (Diff Summarizer).
Your job is to read a unified diff/patch and produce a terse, technical summary
of what changed at the code level. Ignore business intent and high-level context.
Focus on syntax-level modifications only.

INPUT: A .diff or .patch content.
OUTPUT: A bullet list (in English) enumerating concrete code changes, e.g.:
- "Added a new parameter `redirect_url` to function `handle_login`."
- "Removed call to `sanitize_input()` for variable `username`."
- "Introduced new conditional branch (if/else) validating `redirect_url`."
- "Renamed variable `pwd` to `password` in `auth.py`."

Guidelines:
- Be specific: mention function/method names, parameters, files, and constructs (if/else, try/except, loops).
- Mention additions, deletions, renames, signature changes, visibility changes, imports, and modified return values.
- If tests or config changed, note them explicitly.
- Do not speculate on business purpose or user impact.
- Keep it concise and scannable.
"""

SYSTEM_PROMPT_VULN = """
You are a specialized Security Analyst called "Vulnerability Hunter".
You are primed with OWASP Top 10 and CWE knowledge. Your job is to
review:
- Code context summary (from Agent 1),
- Technical diff summary (from Agent 2),
- The raw code diff (added/modified lines),

and produce a concise list of potential vulnerabilities with reasoning.

OUTPUT REQUIREMENTS:
- Write in English.
- Use bullet points. For each item include:
  * Name of vulnerability and (CWE-xxx) if known.
  * Very short justification referencing concrete signals:
    - parameters/fields introduced or modified,
    - removed validations/sanitization,
    - risky APIs, string interpolation, redirects, cookie/session use,
    - auth/authorization checks, input parsing, serialization/deserialization,
    - file/OS operations, crypto, error handling, logging.
  * (Optional) brief risk context if Agent 1 indicates sensitive flows (e.g., login, payments).
- If no clear risk is found, say: "No obvious vulnerabilities found based on provided diffs."

STRICTLY AVOID:
- Business speculation.
- Repeating the entire diff.
- Vague statements without pointing to specific change symptoms.

FORMAT EXAMPLE:
- Potential Vulnerability: Open Redirect (CWE-601). Reason: New `redirect_url` parameter (from Agent 2) is used for a redirect without validation (see added lines). Agent 1 indicates login flow, so impact is high.
- Potential Vulnerability: SQL Injection (CWE-89). Reason: `sanitize_input()` call removed for `username`; value is used to build a DB query.

Be skeptical, but concise and specific.
"""


# =========================
# Konfiguracja
# =========================

@dataclass(frozen=True)
class AgentConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    api_key: str | None = None  # domyślnie weź z ENV


# =========================
# Klasa agregująca agentów
# =========================

class CodeAgents:
    """Prosta fasada do dwóch łańcuchów: analizatora kontekstu i podsumowania diffów."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        load_dotenv()
        api_key = (config.api_key if config else None) or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set it in your environment or .env file."
            )

        self._config = config or AgentConfig(api_key=api_key)

        # Zbuduj łańcuchy
        self._context_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT,
            human_template="Here is the source code to analyze:\n\n```{input_code}```",
        )
        self._diff_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_DIFF,
            human_template="Here is the diff/patch to summarize technically:\n\n```diff\n{diff_content}\n```",
        )

        self._vuln_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_VULN,
            human_template=(
                "Based on the following inputs, identify potential vulnerabilities:\n\n"
                "Code Context Summary:\n{context_summary}\n\n"
                "Diff Summary:\n{diff_summary}\n\n"
                "Raw Diff Content:\n```diff\n{diff_content}\n```"
            ),
        )

    def _make_chain(self, *, system_prompt: str, human_template: str):
        llm = ChatGoogleGenerativeAI(
            model=self._config.model,
            temperature=self._config.temperature,
            api_key=self._config.api_key,  # jawnie, choć i tak czyta z ENV
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_template)]
        )
        return prompt | llm | StrOutputParser()

    # ============ Publiczny interfejs ============
    def analyze_code(self, input_code: str) -> str:
        """Zwraca 2–3 akapity podsumowania kontekstu kodu."""
        return self._context_chain.invoke({"input_code": input_code})

    def summarize_diff(self, diff_content: str) -> str:
        """Zwraca zwięzłą listę punktów opisującą zmiany w diffie."""
        return self._diff_chain.invoke({"diff_content": diff_content})
    
    def find_vulnerabilities(
        self, context_summary: str, diff_summary: str, diff_content: str) -> str:
        """Zwraca listę potencjalnych luk bezpieczeństwa na podstawie podsumowań i diffu."""
        return self._vuln_chain.invoke(
            {
                "context_summary": context_summary,
                "diff_summary": diff_summary,
                "diff_content": diff_content,
            }
        )


# =========================
# Przykład użycia
# =========================

SAMPLE_CODE = """
from flask import Flask, request, jsonify, session
from werkzeug.security import check_password_hash
from db_models import User
from db import get_db_connection

app = Flask(__name__)
app.secret_key = 'very_secret_session_key' # Never do this in production!

@app.route('/login', methods=['POST'])
def handle_login():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing login data"}), 400

    username = data['username']
    password = data['password']

    db = get_db_connection()
    user = db.query(User).filter_by(username=username).first()

    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid credentials"}), 401

    # Set the session after a successful login
    session['user_id'] = user.id
    session['username'] = user.username
    
    return jsonify({"message": f"Welcome, {user.username}!"}), 200
"""

SAMPLE_DIFF = """diff --git a/app.py b/app.py
index 1a2b3c4..5d6e7f8 100644
--- a/app.py
+++ b/app.py
@@ -1,6 +1,7 @@
-from flask import Flask, request, jsonify, session
+from flask import Flask, request, jsonify, session, redirect
 from werkzeug.security import check_password_hash
 from db_models import User
 from db import get_db_connection
+from urllib.parse import urlparse
@@
-@app.route('/login', methods=['POST'])
-def handle_login():
+@app.route('/login', methods=['POST'])
+def handle_login(redirect_url: str | None = None):
     data = request.get_json()
-    if not data or 'username' not in data or 'password' not in data:
+    if not data or 'username' not in data or 'password' not in data:
         return jsonify({"error": "Missing login data"}), 400
@@
-    username = data['username']
-    password = data['password']
+    username = data['username']
+    password = data['password']
@@
-    if not user or not check_password_hash(user.password_hash, password):
+    if not user or not check_password_hash(user.password_hash, password):
         return jsonify({"error": "Invalid credentials"}), 401
@@
-    session['user_id'] = user.id
-    session['username'] = user.username
-    return jsonify({"message": f"Welcome, {user.username}!"}), 200
+    session['user_id'] = user.id
+    session['username'] = user.username
+    # If a safe redirect_url is provided, redirect after login
+    if redirect_url:
+        parsed = urlparse(redirect_url)
+        if parsed.scheme in ('http', 'https'):
+            return redirect(redirect_url, code=302)
+    return jsonify({"message": f"Welcome, {user.username}!"}), 200
"""

if __name__ == "__main__":
    print("--- Initializing agents ---")
    agents = CodeAgents()  # GEMINI_API_KEY z .env lub ENV

    print("\n--- Running analysis ---\n")
    analyzed_code = agents.analyze_code(SAMPLE_CODE)
    summarized_diff = agents.summarize_diff(SAMPLE_DIFF)
    vulnerabilities = agents.find_vulnerabilities(
        analyzed_code, summarized_diff, SAMPLE_DIFF
    )

    print("\n--- Code Context Summary ---\n")
    print(analyzed_code)
    print("\n--- Diff Summary ---\n")
    print(summarized_diff)
    print("\n--- Potential Vulnerabilities ---\n")
    print(vulnerabilities)
