from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the models with your API keys
# (It's best practice to store these as environment variables)
#llm_chatgpt = ChatOpenAI(model_name="gpt-4", openai_api_key="YOUR_OPENAI_API_KEY")
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
#llm_claude = ChatAnthropic(model="claude-3-opus-20240229", anthropic_api_key="YOUR_ANTHROPIC_API_KEY")

# Now you can use them interchangeably
question = "What are the three most important things to learn in programming?"

#response_chatgpt = llm_chatgpt.invoke(question)
response_gemini = llm_gemini.invoke(question)
#response_claude = llm_claude.invoke(question)

#print("ChatGPT's Answer:", response_chatgpt.content)
print("Gemini's Answer:", response_gemini.content)
#print("Claude's Answer:", response_claude.content)