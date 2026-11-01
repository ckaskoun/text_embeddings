"""
Builds a small, fully synthetic copy of the notebook's data folder.

No real data is used: every sentence is generated text, and every embedding is a
random vector. The files have the same names, sheets, and columns as the real
`raw_data/` and `saved_embeddings/` files, so the notebook can run end to end.

Usage:
    python tests/make_fixtures.py <output_folder>
"""

import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# These must match the Setup cell of the notebook.
EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-8B'
EMBEDDING_REVISION = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'
MAX_SEQ_LENGTH = 512
DIMENSIONS = 64

rng = np.random.default_rng(0)


def code_vectors(codes):
    """One random direction per code, so sentences with a code cluster together."""
    return {code: rng.normal(size=DIMENSIONS) for code in codes}


def embed(texts, labels, directions, noise=1.6):
    """
    Returns one random vector per text: the sum of its codes' directions plus noise.
    <labels>: list of lists of codes, one list per text.
    """
    vectors = []
    for codes in labels:
        center = sum((directions[c] for c in codes), np.zeros(DIMENSIONS))
        vectors.append(center + noise * rng.normal(size=DIMENSIONS))
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def save_embeddings(folder, dataset, texts, vectors):
    """Saves vectors in the notebook's .pkl format (no prompt)."""
    unique = pd.Series(texts).drop_duplicates()
    name = f"{EMBEDDING_MODEL.replace('/', '_')}_{dataset}_no_prompt.pkl"
    with open(folder / 'saved_embeddings' / name, 'wb') as f:
        pkl.dump({
            'model': EMBEDDING_MODEL, 'revision': EMBEDDING_REVISION,
            'prompt': '', 'max_length': MAX_SEQ_LENGTH,
            'sentences': unique.tolist(), 'embeddings': vectors[unique.index.to_numpy()]
        }, f)


def make_sources(folder):
    """Sources: one code per sentence (L, O, P, S), plus a human bank and C/D labels."""
    codes = ['L', 'O', 'P', 'S']
    directions = code_vectors(codes)
    weights = [.55, .2, .13, .12]

    corpus_codes = rng.choice(codes, size=400, p=weights)
    corpus = pd.DataFrame({
        'response': [f'Sources response {i} ({c})' for i, c in enumerate(corpus_codes)],
        'updated_code': corpus_codes
    })
    # Repeat a few sentences, as in the real corpus.
    corpus = pd.concat([corpus, corpus.iloc[:12]], ignore_index=True)

    bank_codes = np.repeat(codes, [10, 9, 6, 7])
    bank = pd.DataFrame({
        'text': [f'Sources exemplar {i} ({c})' for i, c in enumerate(bank_codes)],
        'code': bank_codes
    })
    # Two bank sentences also appear in the corpus.
    bank.loc[0, 'text'] = corpus.loc[corpus['updated_code'].eq('L'), 'response'].iloc[0]
    bank.loc[10, 'text'] = corpus.loc[corpus['updated_code'].eq('O'), 'response'].iloc[0]

    # Every bank sentence gets a central (C) or diverse (D) label.
    typed = bank.copy()
    typed['Final'] = np.where(np.arange(len(typed)) % 3 == 0, 'C', 'D')

    corpus.to_excel(folder / 'raw_data' / 'sources_v2.xlsx', index=False)
    bank.to_excel(folder / 'raw_data' / 'centroids_expanded.xlsx', index=False)
    typed.to_excel(folder / 'raw_data' / 'human_central-diverse.xlsx', index=False)

    texts = pd.concat([corpus['response'], bank['text']], ignore_index=True)
    labels = [[c] for c in corpus['updated_code']] + [[c] for c in bank['code']]
    save_embeddings(folder, 'sources_v2', texts, embed(texts, labels, directions, noise=4.0))


def make_pepper(folder):
    """PEPPER: multi-label sentences (DRR, FLS, URR, S3, S5, S7) and a typed human bank."""
    subcodes = ['DRR', 'FLS', 'URR', 'S3', 'S5', 'S7']
    directions = code_vectors(subcodes)

    corpus = pd.DataFrame(
        (rng.random((500, len(subcodes))) < [.12, .1, .1, .12, .12, .1]).astype(int),
        columns=subcodes
    )
    corpus.insert(0, 'Sentence', [f'PEPPER sentence {i}' for i in range(len(corpus))])

    long_names = {
        'DRR': 'Did real research', 'FLS': 'Felt like a scientist',
        'URR': 'Understanding real research', 'S3': 'Failure',
        'S5': 'Iteration', 'S7': 'Decision making'
    }
    sheets = {}
    for code, name in long_names.items():
        types = ['Typical'] * 4 + ['Atypical'] * 2 + ['Close but no'] * 2
        sheets[name] = pd.DataFrame({
            'exemplar_text': [f'PEPPER exemplar {code} {i}' for i in range(len(types))],
            'code': name, 'exemplar_type': types
        })
    # One sentence recorded under two codes, and one code the analysis does not use.
    sheets['Failure'].loc[0, 'exemplar_text'] = sheets['Iteration'].loc[0, 'exemplar_text']
    sheets['Other'] = pd.DataFrame({
        'exemplar_text': ['PEPPER exemplar unused 0'], 'code': 'Unused code',
        'exemplar_type': 'Typical'
    })

    with pd.ExcelWriter(folder / 'raw_data' / 'coding_master_alt_final.xlsx') as writer:
        corpus.to_excel(writer, sheet_name='MAIN', index=False)
    with pd.ExcelWriter(folder / 'raw_data' / 'exemplar_storage.xlsx') as writer:
        for name, sheet in sheets.items():
            sheet.to_excel(writer, sheet_name=name[:31], index=False)

    bank = pd.concat(sheets.values(), ignore_index=True)
    short = {v: k for k, v in long_names.items()}
    corpus_labels = [[c for c in subcodes if row[c] == 1] for _, row in corpus.iterrows()]
    bank_labels = [
        [short[row.code]] if row.exemplar_type != 'Close but no' and row.code in short else []
        for row in bank.itertuples()
    ]
    texts = pd.concat([corpus['Sentence'], bank['exemplar_text']], ignore_index=True)
    save_embeddings(folder, 'pepper', texts, embed(texts, corpus_labels + bank_labels, directions))


def main(folder):
    folder = Path(folder)
    (folder / 'raw_data').mkdir(parents=True, exist_ok=True)
    (folder / 'saved_embeddings').mkdir(parents=True, exist_ok=True)
    make_sources(folder)
    make_pepper(folder)
    print(f'Synthetic fixtures written to {folder}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
