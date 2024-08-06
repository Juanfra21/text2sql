import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

class SQLGenerator:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_sql(self, prompt, schema):
        input_text = "translate English to SQL: " + prompt + " " + schema
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        max_output_length = 1024
        outputs = self.model.generate(**inputs, max_length=max_output_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_path = "path/to/t5_sql_model"
    generator = SQLGenerator(model_path)

    print("Enter 'quit' to exit.")
    while True:
        prompt = input("Insert prompt: ")
        if prompt.lower() == 'quit':
            break
        schema = input("Insert schema: ")

        sql_query = generator.generate_sql(prompt, schema)
        print(f"Generated SQL query: {sql_query}")
        print()

if __name__ == "__main__":
    main()