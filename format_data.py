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
    <df_copy>: A copy of df_coded containing a higher level of codes.
    """
    cell_idx = 0
    df_copy = df_coded.copy()
    for cell in df_copy['Codes']:
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
        df_copy.at[cell_idx, 'Codes'] = higher_codes
        cell_idx += 1
    return df_copy


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


def weight_coded(df_coded, level='category'):
    """
    Replaces the human-assigned codes with weights for model training purposes.

    Arguments:
    <df_coded>: A combined df of coded memos, with one column for all occurences
                of codes. At default level (not raised to a higher level)
    <level>: The level at which the data should be weighted. Options are:
             'local_global', 'topic', and 'category'. Default is category.

    Returns:
    <df_weighted>: A new df which contians weighted values for each category
                   in place of human assigned codes.
    """
    df_duplicate = df_coded.copy()
    if level == 'local_global':
        categories = []
        for k1 in codebook:
            for k2 in codebook[k1]:
                for code in codebook[k1][k2]:
                    categories.append(code)
    elif level == 'topic':
        df_duplicate = higher_level(df_duplicate)
        categories = []
        for k in codebook:
            categories += list(codebook[k].keys())
    elif level == 'category':
        df_duplicate = higher_level(higher_level(df_duplicate))
        categories = list(codebook.keys())

    df_weights = pd.DataFrame(columns=categories)
    row_idx = 0
    for cell in df_duplicate['Codes']:
        total_codes = len(cell)
        for c in categories:
            if c in cell:
                df_weights.at[row_idx, c] = 1 / total_codes
            else:
                df_weights.at[row_idx, c] = 0.0
        row_idx += 1

    df_weighted = pd.concat([df_duplicate, df_weights], axis=1)

    return df_weighted
