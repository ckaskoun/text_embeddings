import pandas
import re

def clean_documents(path):
    """
    Cleans .xlsx files for use with LLMs

    Arguments:
    <path>: path to a .xlsx file

    Returns a pandas df with one column, 'Sentences', with indices matching line
    numbers in the .xlsx file
    """
    cleaned_docs = {}
    doc_idx = 0

    for path in docs:
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
