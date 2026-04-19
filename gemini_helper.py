import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODELOS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-pro"
]

def analizar(prompt):
    ultimo_error = None

    for nombre in MODELOS:
        try:
            model = genai.GenerativeModel(nombre)
            r = model.generate_content(prompt)
            return r.text
        except Exception as e:
            ultimo_error = e

    return f"Error IA: {ultimo_error}"