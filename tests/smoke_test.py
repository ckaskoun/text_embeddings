"""
Runs the notebook end to end on synthetic data (see make_fixtures.py).

Each configuration edits a few Setup values (dataset, selection mode, and much
smaller repetition counts), skips the %pip cells, and executes every cell.
Nothing here reads the real data folders.

Usage:
    python tests/smoke_test.py [notebook] [--configs NAME ...] [--save-results DIR]

Requires: nbformat, nbclient, ipykernel, plus the notebook's own packages.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).parent))
import make_fixtures  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Setup values to change for each run. Values are Python source text.
FAST = {'n_repetitions': '3', 'exemplar_counts': 'list(range(1, 16))', 'use_saved': 'True'}
CONFIGS = {
    'sources_human': {'coding_scheme': 'sources_base', 'exemplar_selection_source': "'human'"},
    'sources_machine': {'coding_scheme': 'sources_base', 'exemplar_selection_source': "'machine'"},
    'sources_combined': {'coding_scheme': 'sources_combined', 'exemplar_selection_source': "'human'"},
    'pepper': {'coding_scheme': 'PEPPER_combined', 'exemplar_selection_source': "'machine'"},
}

# Variables collected after each run, so two notebook versions can be compared.
RESULT_NAMES = [
    'results', 'full_corpus_results', 'calibration', 'selection_summary', 'near_best',
    'df_similarities', 'n_subcentroids', 'main_cutoffs', 'selection_results'
]


def set_value(source, name, value):
    """Replaces the first top-level `name = ...` line in a cell's source."""
    pattern = re.compile(rf'^{name} = .*$', re.MULTILINE)
    if not pattern.search(source):
        raise ValueError(f'Setting {name} was not found in the Setup cell.')
    return pattern.sub(f'{name} = {value}', source, count=1)


def prepare(notebook, data_folder, config, results_file=None):
    """Returns a copy of the notebook edited for one synthetic run."""
    nb = nbformat.from_dict(notebook)
    for cell in nb.cells:
        tags = cell.get('metadata', {}).get('tags', [])
        if cell.cell_type != 'code':
            continue
        if 'install_packages' in tags or 'install_embedding_packages' in tags:
            cell.source = 'pass  # Skipped in the smoke test'
        if 'settings' in tags:
            source = set_value(cell.source, 'local_folder', f'Path({str(data_folder)!r})')
            for name, value in {**FAST, **CONFIGS[config]}.items():
                source = set_value(source, name, value)
            cell.source = source

    if results_file is not None:
        collect = (
            'import pickle as _pkl\n'
            f'_names = {RESULT_NAMES!r}\n'
            f'with open({str(results_file)!r}, "wb") as _f:\n'
            '    _pkl.dump({n: globals()[n] for n in _names if n in globals()}, _f)\n'
        )
        nb.cells.append(nbformat.v4.new_code_cell(collect))
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('notebook', nargs='?', default=REPO / 'text_embeddings_FA26.ipynb')
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument('--save-results', type=Path, default=None,
                        help='Folder to save each run\'s key result tables (pickled).')
    parser.add_argument('--save-executed', type=Path, default=None,
                        help='Folder to save each executed notebook, to look at the graphs. '
                             'These contain synthetic data only.')
    args = parser.parse_args()

    notebook = nbformat.read(args.notebook, as_version=4)
    for folder in [args.save_results, args.save_executed]:
        if folder:
            folder.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        make_fixtures.main(tmp)
        for config in args.configs:
            results_file = args.save_results / f'{config}.pkl' if args.save_results else None
            nb = prepare(notebook, tmp, config, results_file)
            print(f'Running {config} ...', flush=True)
            NotebookClient(nb, timeout=900, kernel_name='python3',
                           resources={'metadata': {'path': tmp}}).execute()
            print(f'  {config}: all {len(nb.cells)} cells ran without errors.')
            if args.save_executed:
                nbformat.write(nb, args.save_executed / f'{config}.ipynb')


if __name__ == '__main__':
    main()
