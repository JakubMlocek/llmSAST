import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Load environment variables ---
# Ensure your .env has: GOOGLE_API_KEY=your_gemini_key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Set it in your environment or .env file."
    )

# --- Agent Configuration ---

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

def create_context_analysis_agent():
    """
    Creates and returns the LangChain (LCEL) chain for the context analysis agent,
    backed by Gemini 2.5 via langchain-google-genai.
    """
    # Gemini 2.5 (flash) is fast and cost-efficient for this task.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,           # deterministic/analytical
        api_key=GEMINI_API_KEY # optional; uses env var by default
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Here is the source code to analyze:\n\n```{input_code}```")
    ])

    output_parser = StrOutputParser()

    # LCEL: prompt -> model -> parse
    agent_chain = prompt_template | llm | output_parser
    return agent_chain

# --- Example Usage ---

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

if __name__ == "__main__":
    print("--- Creating Context Analysis Agent (Gemini 2.5) ---")
    context_agent = create_context_analysis_agent()

    print("\n--- Analyzing Sample Code ---")
    print("Input Code (snippet):")
    print('\n'.join(SAMPLE_CODE.split('\n')[:5]) + "\n...")

    try:
        analysis_result = context_agent.invoke({"input_code": SAMPLE_CODE})
        print("\n--- Agent Analysis Result ---")
        print(analysis_result)
    except Exception as e:
        print(f"\nAn error occurred while invoking the agent: {e}")
        print("Please verify your GEMINI_API_KEY and network access.")
