# -*- coding: utf-8 -*-
"""eda_analysis

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15kc7_DSRcqqXK5W1FZLD9yYK9ZlfgYsv
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re

# Load a small subset (20%) for training
dataset = load_dataset('b-mc2/sql-create-context', split="train")
subset_size = int(len(dataset) * 0.2)
dataset = dataset.select(range(subset_size))

# Convert to pandas DataFrame for easier analysis
df = pd.DataFrame(dataset)
print(dataset.features)

# Basic Information
print("Number of rows and columns:", len(dataset), "rows,", len(dataset[0]), "columns")
print("Features (columns):", dataset.column_names)

# Preview the Data
print("\nSample rows:")
print(df.head())

# Data Types and Missing Values
print("\nData types and missing values:")
print(df.info())
print("\nMissing values count per column:")
print(df.isnull().sum())

# Text Standardization: Lowercase all text and remove special characters
def standardize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['question'] = df['question'].apply(standardize_text)
df['context'] = df['context'].apply(standardize_text)

print("Cleaned and standardized data:")
print(df.head())

# Distribution of Categorical Variables
categorical_columns = ['answer', 'question', 'context']
for col in categorical_columns:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts().head(10))  # Display top 10 most frequent values

    # Plotting bar chart for top 10 most frequent values
    top_values = df[col].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_values.index, y=top_values.values)
    plt.title(f"Top 10 values for {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()



# Feature Engineering Ideas
# Example: Word count for 'question' and 'context'
df['question_length'] = df['question'].apply(lambda x: len(x.split()))
df['context_length'] = df['context'].apply(lambda x: len(x.split()))

# Print out the modified DataFrame to see the new features
print("\nDataFrame with new features:")
print(df.head())

# Summary statistics for the new features
print("\nSummary statistics for new features:")
print(df[['question_length', 'context_length']].describe())

# Word Cloud for 'question' column
question_text = ' '.join(df['question'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(question_text)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Questions')
plt.show()

# Box Plot for outlier detection
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['question_length', 'context_length']])
plt.title('Box Plot for Question and Context Length')
plt.show()

# Data Normalization
scaler = MinMaxScaler()
df[['question_length_scaled', 'context_length_scaled']] = scaler.fit_transform(df[['question_length', 'context_length']])
print(df[['question_length', 'context_length', 'question_length_scaled', 'context_length_scaled']].head())

# Heatmap for feature correlation
plt.figure(figsize=(10, 6))
correlation_matrix = df[['question_length', 'context_length']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Derived Features
df['question_word_count'] = df['question'].apply(lambda x: len(x.split()))
print(df[['question', 'question_word_count']].head())

from transformers import T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Tokenize the text data
df['question_tokens'] = df['question'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=128))
df['context_tokens'] = df['context'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=128))

print("Tokenized data:")
print(df[['question', 'question_tokens', 'context', 'context_tokens']].head())

