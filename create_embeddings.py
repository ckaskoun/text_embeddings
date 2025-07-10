# Importing the libraries
# from transformers import AutoModel
import pickle as pkl
import numpy as np
import os
import pandas
from extract_sentences import clean_documents

# Import embeddings model
# model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)

# Clean data in files
filenames = []
folder_path = 'C:/Users/carlo/OneDrive/Documents/memos_excel'
for file in os.listdir(folder_path):
    path = os.path.join(folder_path, file)
    filenames.append(path)

df = clean_documents(filenames)
print(df)

def encode_text(row):
    text = row['Sentences']
    embedding = model.encode(text)
    return embedding.tolist()

# Apply the function along the rows and assign the result to the new 'embedding' column
# df['embedding'] = df.apply(encode_text, axis=1)
