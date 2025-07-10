# Importing the libraries
from transformers import AutoModel
import pickle as pkl
import numpy as np

# Import embeddings model
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
