import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # Załaduj zmienne środowiskowe z pliku .env


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Wklej tutaj swój klucz API
genai.configure(api_key=GEMINI_API_KEY) 

# Utwórz model
model = genai.GenerativeModel('gemini-2.5-flash') 

# Wyślij prompt i pobierz odpowiedź
response = model.generate_content("Jak wykorzystac wiele modeli LLM w jednym projekcie Python?")

# Wyświetl odpowiedź
print(response)