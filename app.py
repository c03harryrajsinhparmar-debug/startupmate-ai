# StartupMate AI - Backend (Python + Flask + Gemini API)
import os
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

app = Flask(__name__)

# The API key is read from an environment variable (never write it in code)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# The app tries these models in order. If one is busy, it uses the next one.
MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"]

# This tells the AI what its job is
SYSTEM_PROMPT = """You are StartupMate AI, a friendly startup idea assistant.
You help users to:
1. Develop and improve their startup ideas.
2. Find their target customers.
3. Create a basic business plan (problem, solution, customers,
   revenue model, costs, and first steps).
Rules: Use simple English. Keep answers short and clear.
Use small bullet points. Ask one helpful question at the end.
If the question is not about startups, politely bring the user back to startups."""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # 'messages' is the full chat history sent from the browser
    messages = request.json.get("messages", [])

    # Gemini uses the role name "model" instead of "assistant"
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    # Try each model one by one until one works
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            return jsonify({"reply": response.text})
        except Exception as e:
            print("Model", model, "failed:", e)   # shows the error in the terminal

    return jsonify({"reply": "Sorry, the AI is busy right now. Please try again in a minute."}), 500

if __name__ == "__main__":
    app.run(debug=False)