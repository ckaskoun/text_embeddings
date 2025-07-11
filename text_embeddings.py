# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy as sc
import math
from create_embeddings import embed_files

# Special imports for plots
import string
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors

df_embeddings = pd.DataFrame(columns=['File','Lines', 'Sentences', 'Embeddings'])

dict_embeddings = embed_files("C:/Users/carlo/OneDrive/Documents/memos_excel")
for k in dict_embeddings:
    if 'Codes' in dict_embeddings[k].columns:
        dict_embeddings[k].drop('Codes', axis=1, inplace=True)
    df = dict_embeddings[k]
    df.insert(loc=0, column='File', value=k)
    df_embeddings = pd.concat([df_embeddings, df], ignore_index=True, join='inner')
