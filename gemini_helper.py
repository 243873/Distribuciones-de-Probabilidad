import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

def analizar_ia(stats, decision_estudiante):

    prompt = f"""
Actúa como profesor universitario de estadística.

n={stats['n']}
media={stats['media']:.4f}
mu0={stats['mu0']}
sigma={stats['sigma']}
alpha={stats['alpha']}
tipo={stats['tipo']}
z={stats['z']:.4f}
p={stats['p']:.6f}

Decisión sistema: {stats['decision']}
Decisión estudiante: {decision_estudiante}

Responde breve:
1. ¿Se rechaza H0?
2. ¿El estudiante acertó?
3. Explica usando Z y p-value.
4. ¿Supuestos razonables?
5. Conclusión práctica.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")

    r = model.generate_content(prompt)

    return r.text