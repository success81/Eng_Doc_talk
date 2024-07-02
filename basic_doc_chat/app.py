from flask import Flask, request, jsonify, render_template
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
import os
import fitz  # PyMuPDF

app = Flask(__name__)

# Ensure the environment variable for Google Cloud credentials is set
if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

# Initialize Vertex AI
vertexai.init(project="clear-safeguard-420323", location="us-central1")
model = GenerativeModel("gemini-1.5-flash-001")

# Store the conversation history and documents
conversation_history = []
documents = {}

def extract_text_from_pdf(file_stream):
    file_stream.seek(0)
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/')
def index():
    global documents, conversation_history
    documents = {}  # Reset the documents each time the page is refreshed
    conversation_history = []  # Reset the conversation history each time the page is refreshed
    return render_template('index.html', documents=documents.keys())

@app.route('/upload', methods=['POST'])
def upload():
    global documents

    if 'files[]' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('files[]')
    if len(files) + len(documents) > 4:
        return jsonify({"error": "You can upload up to 4 files"}), 400

    for file in files:
        filename = file.filename
        if filename in documents:
            continue
        try:
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            else:
                return jsonify({"error": "Unsupported file type. Only PDFs are allowed."}), 400
            documents[filename] = text
            print(f"Extracted text from {filename}: {text[:500]}...")  # Log first 500 characters of extracted text
        except Exception as e:
            print(f"Error extracting text from {filename}: {str(e)}")
            return jsonify({"error": f"Error extracting text from {filename}"}), 500

    return jsonify({"message": "Files uploaded successfully", "documents": list(documents.keys())})

@app.route('/delete', methods=['POST'])
def delete():
    global documents

    filename = request.form.get('filename')
    if filename in documents:
        del documents[filename]
        return jsonify({"message": f"Deleted {filename}", "documents": list(documents.keys())})
    else:
        return jsonify({"error": f"Document {filename} not found"}), 404

@app.route('/clear', methods=['POST'])
def clear():
    global documents, conversation_history
    documents = {}
    conversation_history = []
    return jsonify({"message": "All documents and conversation history cleared", "documents": [], "conversation_history": []})

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history, documents

    try:
        # Get the user input
        user_input = request.form.get("message", "")
        if not user_input:
            raise ValueError("No message provided in the request.")

        # Add the user input to the conversation history
        conversation_history.append(f"User: {user_input}")

        # Keep only the last 20 questions
        conversation_history = conversation_history[-20:]

        # Create the prompt
        documents_text = "\n".join([f"Document {i+1} ({name}):\n{text}" for i, (name, text) in enumerate(documents.items())])
        prompt = f"You are operating as part of a program. The text you should be answering on is uploaded by a user.\n\n{documents_text}\n" + "\n".join(conversation_history)
        print(f"Generated prompt: {prompt[:500]}...")  # Log first 500 characters of the prompt

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

        # Keep only the last 20 messages
        conversation_history = conversation_history[-20:]

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
    app.run(host="0.0.0.0", port=8080)





