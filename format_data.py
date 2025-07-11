import pandas
from create_embeddings import clean_documents

def convert_codes(row):
    """
    """
    code = row['Codes']
    code_lst = code.split(',')
    for i in range(len(code_lst)):
        code_lst[i] = code_lst[i].strip()
    return code_lst

def format_coded(folder_path):
    """
    """
    df_dict = clean_documents(folder_path)
    for k in df_dict:
        df_dict[k]['Codes'] = df_dict[k].apply(convert_codes, axis=1)
    print(df_dict)

def combine_coded(cleaned_df):
    """
    """
    df_coded = pd.DataFrame(columns=['File','Lines', 'Sentences', 'Codes'])

    for k in cleaned_df:
        df = cleaned_df[k]
        df.insert(loc=0, column='File', value=k)
        df_coded = pd.concat([df_coded, df], ignore_index=True, join='inner')

    return df_coded

def combine_embedded(df_dict):
    """
    """
    df_embeddings = pd.DataFrame(columns=['File','Lines', 'Sentences', 'Embeddings'])

    for k in df_dict:
        if 'Codes' in df_dict[k].columns:
            dict_embeddings[k].drop('Codes', axis=1, inplace=True)
        df = dict_embeddings[k]
        df.insert(loc=0, column='File', value=k)
        df_embeddings = pd.concat([df_embeddings, df], ignore_index=True, join='inner')

    return df_embeddings

def sum_codes():


if __name__ =="__main__":
    format_coded("C:/Users/carlo/Downloads/coded_memos") # debug
