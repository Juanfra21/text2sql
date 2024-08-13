# Fine-Tuned Google T5 Model for Text to SQL Translation

A fine-tuned version of the Google T5 model, trained for the task of translating natural language queries into SQL statements.

For additional information, [check out the model on Hugging Face](https://huggingface.co/juanfra218/text2sql)

## Model Details

- **Architecture**: Google T5 Base (Text-to-Text Transfer Transformer)
- **Task**: Text to SQL Translation
- **Fine-Tuning Datasets**: 
  - [sql-create-context Dataset](https://huggingface.co/datasets/b-mc2/sql-create-context) 
  - [Synthetic-Text-To-SQL Dataset](https://huggingface.co/datasets/gretelai/synthetic-text-to-sql)

## Features
  - Natural Language to SQL Translation: Converts complex natural language queries into SQL statements with good accuracy.
  - Streamlit Interface: A user-friendly web interface built using Streamlit, allowing users to input queries and receive corresponding SQL outputs instantly.
  - Flask API: A RESTful API built with Flask, enabling easy integration into various applications and services.

## Ongoing Work

Currently working to implement PICARD (Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models) to improve the results of this model. More details can be found in the original [PICARD paper](https://arxiv.org/abs/2109.05093).

## Results

Results are currently being evaluated and will be posted here soon.
