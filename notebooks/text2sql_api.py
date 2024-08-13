from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load the tokenizer and model
model_path = 'text2sql_model_path'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate SQL queries
def generate_sql(prompt, schema):
    input_text = "translate English to SQL: " + prompt + " " + schema
    #print(input_text)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    max_output_length = 1024
    outputs = model.generate(**inputs, max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# API code structure for generating SQL
@app.route('/generate_sql', methods=['POST'])
def generate_sql_endpoint():
    data = request.json
    #print(data)
    prompt = data.get('prompt')
    schema = data.get('schema')

    if not prompt or not schema:
        return jsonify({"error": "Please provide both prompt and schema"}), 400

    sql_query = generate_sql(prompt, schema)
    return jsonify({"sql_query": sql_query})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
