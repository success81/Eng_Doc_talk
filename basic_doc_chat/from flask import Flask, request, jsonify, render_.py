from flask import Flask, request, jsonify, render_template
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

app = Flask(__name__)

# Initialize Vertex AI
vertexai.init(project="clear-safeguard-420323", location="us-central1")
model = GenerativeModel("gemini-1.5-flash-001")

# Store the conversation history
conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history

    try:
        # Get the user input
        user_input = request.form.get("message", "")
        if not user_input:
            raise ValueError("No message provided in the request.")

        # Add the user input to the conversation history
        conversation_history.append(f"User: {user_input}")

        # Keep only the last 30 messages
        conversation_history = conversation_history[-30:]

        # Create the prompt
        prompt = "\n".join(conversation_history)

        # Generate the response
        responses = model.generate_content(
            [prompt],  # Include the prompt in the list
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        response_text = ""
        for response in responses:
            response_text += response.text

        # Add the model's response to the conversation history
        conversation_history.append(f"AI: {response_text}")

        # Keep only the last 30 messages
        conversation_history = conversation_history[-30:]

        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
