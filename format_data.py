import pandas as pd
from create_embeddings import clean_documents

# Code hierarchy
codebook = {
    "Direct Authenticity": {
        'Did real research' : ['RR_G'],
        'Felt like a scientist': ['FS_G'],
        'Understanding real research': ['UR_G']
    },
    "Indirect Authenticity": {
        'Relevant discovery': ['RD_L', 'RD_G'],
        'Societal impact': ['SI_L', 'SI_G'],
        'Publishable science': ['PS_L', 'PS_G'],
        'External Collaboration': ['EC_L', 'EC_G']
    },
    "Research Components":{
        'Failure': ['F_L', 'F_G'],
        'Comparison to prior studies': ['C_L', 'C_G'],
        'Iteration': ['I_L', 'I_G'],
        'Scientific practices': ['SP_L', 'SP_G'],
        'Decision making': ['DM_L', 'DM_G']
    },
    "No Code": {"No Code": ['NC']}
}

def convert_codes(row):
    """
    Converts codes in .xlsx files into a list of codes.

    Arguments:
    <row> for use with pandas df

    Returns:
    List of codes
    """
    code = row['Codes']
    code_lst = code.split(',')
    for i in range(len(code_lst)):
        code_lst[i] = code_lst[i].strip()
    return code_lst


def format_coded(folder_path):
    """
    Formats coded .xlsx files.

    Arguments:
    <folder_path>: Path to a folder containing multiple .xlsx files that heve
                   been hand-coded and added a column for 'Codes'.

    Returns:
    <dict_formatted>: A dictionary with keys as the filename and entries as
                      a pandas df with formatted codes.
    """
    dict_formatted = clean_documents(folder_path)
    for k in dict_formatted:
        dict_formatted[k]['Codes'] = dict_formatted[k].apply(convert_codes, axis=1)
    return dict_formatted


def combine_coded(df_formatted):
    """
    Combines df within a formatted coded df.

    Arguments:
    <df_formatted>: A dictionary with keys as file names and the contents as dfs
                    containing cleaned text. No embeddings have been made.

    Returns:
    <df_coded>: A single pandas df with columns 'File','Lines',
                'Sentences', and 'Codes'. File is for identification and Lines,
                Sentences and Codes is exactly the same as before.
    """
    df_coded = pd.DataFrame(columns=['File','Lines', 'Sentences', 'Codes'])

    for k in df_formatted:
        df = df_formatted[k]
        df.insert(loc=0, column='File', value=k)
        df_coded = pd.concat([df_coded, df], ignore_index=True, join='inner')

    return df_coded


def combine_embedded(df_dict):
    """
    Combines df within a df_dict (result of calling embed_files). Accounts
    for embedding on a hand-coded document by removing the codes.

    Arguments:
    <df_dict>: A dictionary with keys as file names and the contents as dfs
               containing embedded text. Must have 'Lines', 'Sentences',
               and 'Embeddings' as column names.

    Returns:
    <df_embeddings>: A single pandas df with columns 'File','Lines',
                     'Sentences', and 'Embeddings'. Lines, Sentences, and Embeddings
                     remain the same from the df_dict and a column for the
                     key (filename) is added for identification. df indices
                     do not indicate anything.
    """
    df_embeddings = pd.DataFrame(columns=['File','Lines', 'Sentences', 'Embeddings'])

    for k in df_dict:
        if 'Codes' in df_dict[k].columns:
            df_dict[k].drop('Codes', axis=1, inplace=True)
        df = df_dict[k]
        df.insert(loc=0, column='File', value=k)
        df_embeddings = pd.concat([df_embeddings, df], ignore_index=True, join='inner')

    return df_embeddings


def higher_level(df_coded):
    """
    Combines codes into their parent code: L/G -> single code -> category

    Arguments:
    <df_coded>: A dataframe of all files combines with one column for all of the
                "Codes".

    Returns:
    Changes df_coded to contain a higher level of codes.
    """
    cell_idx = 0
    for cell in df_coded['Codes']:
        higher_codes = []
        for i in range(len(cell)):
            for category in codebook:
                if cell[i] in codebook[category]:
                    if category not in higher_codes:
                        higher_codes.append(category)
                else:
                    for code in codebook[category]:
                        if cell[i] in codebook[category][code]:
                            if code not in higher_codes:
                                higher_codes.append(code)
        df_coded.at[cell_idx, 'Codes'] = higher_codes
        cell_idx += 1
    return df_coded


def sum_codes(df_coded):
    """
    Sums occurence of codes within a set of hand-coded files.

    Arguments:
    <df_coded>: A dataframe of all files combines with one column for all of the
                "Codes".

    Returns:
    <code_occurence>: A dictionary with the codes as keys and values as the
                      occurence of said key in the data set.
    """
    code_occurence = {}
    for cell in df_coded['Codes']:
        for code in cell:
            if code in code_occurence:
                code_occurence[code] += 1
            else:
                code_occurence[code] = 1
    return code_occurence
