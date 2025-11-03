# agents.py
import os
import pandas as pd
import sys
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

script_path = os.path.realpath(__file__)
BASE_DIR = os.path.dirname(script_path)
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

def load_prompt_from_file(filename: str) -> str:
    """Wczytuje zawartość pliku z folderu PROMPTS_DIR."""
    file_path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku promptu. Upewnij się, że plik istnieje.", file=sys.stderr)
        print(f"Oczekiwana ścieżka: {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Błąd podczas wczytywania promptu {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

SYSTEM_PROMPT_ONLY_ONE_AGENT = load_prompt_from_file("single_agent.txt")
SYSTEM_PROMPT_CONTEXT = load_prompt_from_file("context_agent.txt")
SYSTEM_PROMPT_VULN = load_prompt_from_file("vuln_hunter_agent.txt")
SYSTEM_PROMPT_RISK_FP = load_prompt_from_file("fp_remover_agent.txt")


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
                if counter >= 1:
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