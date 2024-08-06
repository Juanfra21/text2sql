# Preprocessing function
def preprocess_function(examples):
    inputs = ["translate English to SQL: " + p + " " + c for p, c in zip(examples['prompt'], examples['context'])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['answer'], max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs