import os
import re
import pandas
from transformers import AutoModel

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
            if '\n' in sentence:
                cleaned_sentence = re.sub(r'\n+', ' ', sentence)
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
    embedding = model.encode(text)
    return embedding.tolist()

def embed_files(folder_path, embed_model="jinaai/jina-embeddings-v2-small-en"):
    """
    Embeds text within a .xlsx file.

    Attributes:
    <folder_path>: Path to a folder containing only .xlsx files in the proper format.
    <embed_model>: The model to be used to embed the text. Default is Jina v2.

    Returns:
    <df_dict>: A dictionary containing filenames without extension as keys and
               dataframes for each file contining a 'Sentence' column with text
               and an 'Embeddings' column containing the embeddings.
    """
    # Define variables needed for encode_text function
    global model, column
    model = AutoModel.from_pretrained(embed_model, trust_remote_code=True)
    column = 'Sentences'

    # Clean .xlsx files
    df_dict = clean_documents(folder_path)

    # Apply encode_text function to sentences for each file, update df
    for k in df_dict:
        df = df_dict[k]
        df['Embeddings'] = df.apply(encode_text, axis=1)
        df_dict[k] = df

    return df_dict
