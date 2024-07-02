from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import load_config

def generate_sql(question, schema, model_path='results/text-to-sql-model'):
    config = load_config()

    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    input_text = question + " [schema] " + schema
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=config['max_length'], truncation=True)
    outputs = model.generate(inputs, max_length=config['max_length'], num_beams=4, early_stopping=True)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

if __name__ == "__main__":
    question = "How many houses are there in New York?"
    schema = "CREATE TABLE houses (id INTEGER, address VARCHAR, city VARCHAR)"
    sql_query = generate_sql(question, schema)
    print(sql_query)