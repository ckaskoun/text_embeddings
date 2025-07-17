"""
Create embeddings from sentences. Coded currently for Jina v4 and may need
adjustment for any other models.
"""

# Imports
import os
import re
import pandas
from sentence_transformers import SentenceTransformer

def clean_documents(folder_path):
    """
    Cleans .xlsx files from a folder for use with LLMs.

    Arguments:
    <folder>: path to a folder containing >= 1 .xlsx file with the first two
              columns being numbers 0, 1, 2, ... and text.

    Returns:
    <cleaned_docs>: a dict of pandas dfs with one column, 'Sentences', with
                    indices matching line numbers in the first column of the .xlsx file.
                    Dictionary keys are the file names without extension.
    """
    # Retrieve file paths
    file_names = []
    for file_path in os.listdir(folder_path):
        path = os.path.join(folder_path, file_path)
        file_names.append(path)

    cleaned_docs = {}

    for path in file_names:
        # Find base name
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        df = pandas.read_excel(path)
        # Rename Line Number column & remove newlines
        df.rename(columns={'Unnamed: 0': 'Lines'}, inplace=True)
        for sentence in df['Sentences']:
            try:
                if '\n' in sentence:
                    cleaned_sentence = re.sub(r'\n+', ' ', sentence)
                    cleaned_sentence = cleaned_sentence.strip('\n')
                    df['Sentences'] = df['Sentences'].replace([sentence], cleaned_sentence)
            except:
                if '\n' in str(sentence):
                    cleaned_sentence = re.sub(r'\n+', ' ', str(sentence))
                    cleaned_sentence = cleaned_sentence.strip('\n')
                    df['Sentences'] = df['Sentences'].replace([sentence], cleaned_sentence)
        cleaned_docs[name] = df

    return cleaned_docs

def encode_text(row):
    """
    Reads the text within a pandas df in a specific column and per row,
    creates an embedding using the assigned model.

    > This function requires a pre-defined model and column name
      before it is called.

    Arguments:
    <row>: For applying to each row of a pandas df.

    Returns:
    A list of the coordinates of the embeddings.
    """
    text = row[column]
    if isinstance(text, str):
        embedding = model.encode(text)
    else:
        embedding = model.encode(str(text))
    return embedding.tolist()

def embed_files(folder_path, embed_model="jinaai/jina-embeddings-v4"):
    """
    Embeds text within a .xlsx file.

    Arguments:
    <folder_path>: Path to a folder containing only .xlsx files in the proper format.
    <embed_model>: The model to be used to embed the text. Default is Jina v2.

    Returns:
    <df_dict>: A dictionary containing filenames without extension as keys and
               dataframes for each file contining a 'Sentence' column with text
               and an 'Embeddings' column containing the embeddings.
    """
    # Define variables needed for encode_text function, specific for Jina v4 and may
    # need adjustment otherwise
    global model, column, embed
    try:
        if embed_model != embed:
            model = SentenceTransformer(embed_model, trust_remote_code=True, model_kwargs={"default_task": "text-matching"})
            embed=embed_model
    except:
        model = SentenceTransformer(embed_model, trust_remote_code=True, model_kwargs={"default_task": "text-matching"})
        embed=embed_model
    column = 'Sentences'
    embed=embed_model

    # Clean .xlsx files
    df_dict = clean_documents(folder_path)

    # Apply encode_text function to sentences for each file, update df
    for k in df_dict:
        df = df_dict[k]
        df['Embeddings'] = df.apply(encode_text, axis=1)
        df_dict[k] = df

    return df_dict
