def preprocess_function(examples, tokenizer, max_length=512):
    inputs = [q + " [schema] " + s for q, s in zip(examples['question'], examples['context'])]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')
    labels = tokenizer(examples['answer'], max_length=max_length, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs