import pandas
import re

def clean_documents(filenames):
    """
    Cleans .xlsx files for use with LLMs

    Arguments:
    <filenames>: path to a folder of .xlsx files

    Returns a dict of pandas dfs with one column, 'Sentences', with indices
    matching line numbers in the .xlsx file
    """
    cleaned_docs = {}
    doc_idx = 0

    for path in filenames:
        df = pandas.read_excel(path)
        df.drop(columns='Unnamed: 0', inplace=True)
        for sentence in df['Sentences']:
            if '\n' in sentence:
                cleaned_sentence = re.sub(r'\n+', ' ', sentence)
                cleaned_sentence = cleaned_sentence.strip('\n')
                df['Sentences'] = df['Sentences'].replace([sentence], cleaned_sentence)
        cleaned_docs[f'Doc{doc_idx}'] = df
        doc_idx += 1

    return cleaned_docs
