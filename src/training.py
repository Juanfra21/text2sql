from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from pathlib import Path
from data_processing import load_and_preprocess_data
from utils import preprocess_function
from config import load_config


def train_model(config_path='config/config.yaml'):
    config = load_config(config_path)

    train_dataset, val_dataset = load_and_preprocess_data(
        dataset_name=config['dataset_name'],
        subset_fraction=0.2,
        train_fraction=config['train_size']
    )

    tokenizer = T5Tokenizer.from_pretrained(config['model_name'])
    model = T5ForConditionalGeneration.from_pretrained(config['model_name'])

    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        evaluation_strategy="epoch",
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['num_epochs'],
        weight_decay=config['weight_decay'],
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
    )

    trainer.train()

    path = Path('results')
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained('results/text-to-sql-model')
    tokenizer.save_pretrained('results/text-to-sql-model')

if __name__ == "__main__":
    train_model()