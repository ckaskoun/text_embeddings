# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy as sc
import math
import create_embeddings as ct
import format_data as dt

# Special imports for plots
import string
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors

dict_embeddings = ct.embed_files("C:/Users/carlo/OneDrive/Documents/coded_memos")
df_embeddings = dt.combine_embedded(dict_embeddings)

dict_codes = dt.format_coded("C:/Users/carlo/OneDrive/Documents/coded_memos")
df_codes = dt.combine_coded(dict_codes)
