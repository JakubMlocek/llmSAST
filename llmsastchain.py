# agents.py
import os
import pandas as pd
import sys
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT_ONLY_ONE_AGENT = """
You are a "Code Vulnerability Auditor", a senior software architect with
deep expertise in the OWASP Top 10 and CWE.

Your task is to conduct a complete security review of a code change
provided as a raw 'git diff'.

You will receive the full content of the file (for context) and the
raw 'git diff' showing the changes.

Perform the following steps internally:
1.  **Context Analysis:** Understand the code's main purpose (e.g., login, payments, data processing).
2.  **Change Analysis:** Identify what technically changed (new inputs, modified logic, removed validation).
3.  **Vulnerability Detection:** Based on the context and the change, identify potential vulnerabilities (e.g., SQLi, XSS, IDOR, Open Redirect).
4.  **Risk & FP Assessment:** Critically evaluate your own findings. Reject clear false positives (e.g., if you see sanitization, parameterization, or safe defaults being used).

OUTPUT (English):
- A concise summary of your findings.
- Use a bulleted list for vulnerabilities. For each item, provide:
  * "Confirmed Vulnerability (High/Medium/Low Risk): <name> (CWE-xxx). <Reasoning referencing the diff>."
  * "Rejected (False Positive): <name>. <Reasoning why it is not a risk>."
- If no clear risks are found, state: "No obvious vulnerabilities found."
"""

SYSTEM_PROMPT_CONTEXT = """
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


SYSTEM_PROMPT_VULN = """
You are a specialized Security Analyst called "Vulnerability Hunter".
You are primed with OWASP Top 10 and CWE knowledge. Your job is to
review:
- Code context summary (from Agent 1),

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

SYSTEM_PROMPT_RISK_FP = """
You are Agent: "Risk & FP Verifier" (Weryfikator Ryzyka i Fałszywych Pozytywów).
Act as a devil's advocate to critically evaluate the vulnerability hypotheses produced by the Vulnerability Hunter Agent.


GOAL:
- Reduce false positives by seeking concrete evidence in the provided context that DISCONFIRMS or LOWERS the risk of a vulnerability.
- Confirm only the findings that remain well-supported.


INPUTS:
- A list of potential vulnerabilities from previous agent.
- The code context summary from first Agent.


OUTPUT (English):
- A filtered and assessed list. For each item, choose one of:
* "Confirmed Vulnerability (High/Medium/Low Risk): <name>. <short reasoning>."
* "Rejected (False Positive): <name>. <short reasoning – concrete evidence contradicts the finding>."
* "Inconclusive: <name>. <short reasoning – specify what evidence is missing>."
- Keep each item to 1–3 short sentences; cite concrete signals (parameters, removed checks, API usage, host/domain validation, ORM parameterization, etc.). Avoid repeating large code blocks.


PRINCIPLES:
- Prefer conservative confirmations; when in doubt and lacking evidence, mark as Inconclusive rather than Confirmed.
- If validation, sanitization, or safe defaults are present (e.g., ORM parameterization, allow-list redirects, CSRF protection), use them to reject or lower risk.
- Do not invent system behavior beyond what the inputs support.
"""


@dataclass(frozen=True)
class AgentConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    api_key: str | None = None  # domyślnie weź z ENV



class CodeAgents:

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
            system_prompt=SYSTEM_PROMPT_CONTEXT,
            human_template="Here is the source code to analyze:\n\n```{input_code}```",
        )

        self._vuln_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_VULN,
            human_template=(
                "Based on the following inputs, identify potential vulnerabilities:\n\n"
                "Code Context Summary:\n{context_summary}\n\n"
            ),
        )

        self._risk_fp_chain = self._make_chain(
            system_prompt=SYSTEM_PROMPT_RISK_FP,
            human_template=(
                "Critically evaluate the following potential vulnerabilities:\n\n"
                "Vulnerabilities List:\n{vuln_list}\n\n"
                "Code Context Summary:\n{context_summary}\n\n"
            ),
        )

    def _make_chain(self, *, system_prompt: str, human_template: str):
        llm = ChatGoogleGenerativeAI(
            model=self._config.model,
            temperature=self._config.temperature,
            api_key=self._config.api_key, 
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_template)]
        )
        return prompt | llm | StrOutputParser()

    # ============ Publiczny interfejs ============
    def analyze_code(self, input_code: str) -> str:
        """Zwraca 2–3 akapity podsumowania kontekstu kodu."""
        return self._context_chain.invoke({"input_code": input_code})
    
    def find_vulnerabilities(
        self, context_summary: str) -> str:
        """Zwraca listę potencjalnych luk bezpieczeństwa na podstawie podsumowań i diffu."""
        return self._vuln_chain.invoke(
            {
                "context_summary": context_summary,
            }
        )

    def verify_risk_and_fp(self, *, vuln_list: str, context_summary: str) -> str:
        """Uruchamia Weryfikatora Ryzyka i Fałszywych Pozytywów (Agent 4).
        Przyjmuje wynik Agenta 3 plus kontekst i zwraca przefiltrowaną ocenę."""
        return self._risk_fp_chain.invoke({
        "vuln_list": vuln_list,
        "context_summary": context_summary,        }
    )


def run_short_code_dataset():
    print("--- Initializing agents ---")
    agents = CodeAgents()  
    
    csv_file = "vulnerability_fix_dataset.csv"

    counter = 0
    
    try:
        df = pd.read_csv(csv_file)
        
        if 'vulnerable_code' not in df.columns:
            print(f"Error: Column 'vulnerable_code' not found in {csv_file}", file=sys.stderr)
        else:
            print(f"--- Successfully loaded {len(df)} samples from {csv_file} ---")
            
            for index, row in df.iterrows():
                sample_code = row['vulnerable_code']
                
                # Skip rows where the code is missing
                if not isinstance(sample_code, str) or not sample_code.strip():
                    print(f"\n--- Skipping row {index + 1} (no code found) ---")
                    continue
                
                print(f"\n--- Processing Sample {index + 1} of {len(df)} ---")
                
                print("--- Running analysis ---")
                analyzed_code = agents.analyze_code(sample_code)
                vulnerabilities = agents.find_vulnerabilities(
                    analyzed_code,
                )
                verdict = agents.verify_risk_and_fp(
                    vuln_list=vulnerabilities,
                    context_summary=analyzed_code,
                )   

                # --- Print the results for this specific sample ---
                print("\n--- Code Context Summary ---")
                print(analyzed_code)
                print("\n--- Potential Vulnerabilities ---")
                print(vulnerabilities)
                print("\n--- Final Risk & FP Assessment ---")
                print(verdict)
                print(f"--- Finished Processing Sample {index + 1} ---")

                counter += 1
                if counter >= 3:
                    print("\n--- Reached processing limit of 3 samples. Exiting. ---")
                    break

    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.", file=sys.stderr)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_file} is empty.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_short_code_dataset()