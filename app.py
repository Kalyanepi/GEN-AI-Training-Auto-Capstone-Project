
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from config.settings import OPENAI_API_KEY, MODEL_NAME, ASSISTANT_NAME, SYSTEM_PROMPT
from core.workflows.loan_workflow import LoanEligibilityWorkflow 

app = Flask(__name__)
CORS(app)
client =OpenAI(api_key=OPENAI_API_KEY)

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Add user message
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # OpenAI API call
    response = client.responses.create(
        model=MODEL_NAME,
        input=conversation_history,
    )

    assistant_message = response.output_text

    # Add assistant reply
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return jsonify({
        "response": assistant_message,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    })


@app.route("/clear", methods=["POST"])
def clear():
    global conversation_history
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    return jsonify({"status": "cleared"})


@app.route("/loan-evaluate", methods=["POST"]) 
def loan_evaluate(): 
   data = request.json 
   workflow = LoanEligibilityWorkflow(bot) 
   result = workflow.evaluate(data) 
   return jsonify(result)

@app.route("/test-hallucination", methods=["POST"]) 
def test_hallucination(): 
   query = request.json.get("query") 
   reply, _ = bot.ask(query) 
   return jsonify({"reply": reply}) 

if __name__ == "__main__":
    app.run(debug=True, port=5000)
