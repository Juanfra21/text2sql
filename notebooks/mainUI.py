import streamlit as st
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
model_path = 'text2sql_model_path'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate SQL queries
def generate_sql(prompt, schema):
    input_text = "Translate English to SQL: " + prompt + " " + schema
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    max_output_length = 1024
    outputs = model.generate(**inputs, max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("Text to SQL Generator")
st.write("Enter a natural language query and schema to generate the corresponding SQL.")

prompt = st.text_input("Insert prompt:")
schema = st.text_input("Insert schema:")

if st.button("Generate SQL"):
    if prompt and schema:
        sql_query = generate_sql(prompt, schema)
        st.write("Generated SQL query:")
        st.write(sql_query)
    else:
        st.write("Please enter both a prompt and a schema!")

if st.button("Clear"):
    prompt = ""
    schema = ""
