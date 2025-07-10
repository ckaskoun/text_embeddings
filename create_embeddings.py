# Importing the libraries
from transformers import AutoModel
import pickle as pkl
import numpy as np

# Import embeddings model
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)

# Load dataset
with open("data/PERC2001-2023_ExtraArticles/processed_text.pkl", "rb") as f:
    df = pkl.load(f)
df.head()

%%time
def encode_text(row):
    text = row['raw']
    embedding = model.encode(text)
    return embedding.tolist()

# Apply the function along the rows and assign the result to the new 'embedding' column
df['embedding'] = df.apply(encode_text, axis=1)

with open("../data/PERC2001-2023_ExatraArticles/embeddings_jina.pkl", "wb") as f:
    pkl.dump(df, f)
