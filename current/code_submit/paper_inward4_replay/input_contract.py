"""Current source description and frozen analysis settings."""
import json
from pathlib import Path




def input_provenance():
    """Return the current hospital-confirmed input description."""
    return {'source_status': 'author_confirmed_measured', 'source_basis': 'Hospital experimenter confirmation, as communicated by the study authors.', 'source_statement': 'U-TEG MA and curves are original measured values.', 'runtime_input': 'U-TEG inputs are indexed on a 900-point grid at one-second intervals (15 minutes). The 5- and 10-minute windows cover the first 300 and 600 grid points. Processing masks identify the points retained for fitting. The runner loads saved early fits and stored full-record coefficients; parameter preparation precedes model evaluation.', 'runtime_rescaling': False, 'runtime_reference_labels_used_to_build_features': False}




def check_protocol(saved, expected):
    """Check the frozen analysis schema; source declarations are separate metadata."""
    fields = ('data_sha256', 'source', 'required', 'scopes', 'selection',
              'manuscript_changed')
    metadata = {'B_from_original_data', 'source_status', 'input_provenance'}
    unknown = set(expected) - set(fields) - metadata
    if unknown:
        raise ValueError(f'Unregistered analysis protocol fields: {sorted(unknown)}')
    missing = set(fields) - set(saved) | (set(fields) - set(expected))
    if missing:
        raise ValueError(f'Missing analysis protocol fields: {sorted(missing)}')
    changed = [name for name in fields if saved[name] != expected[name]]
    if changed:
        raise ValueError(f'Analysis protocol changed: {changed}')


def write_input_contract(path):
    Path(path).write_text(json.dumps(input_provenance(), indent=2) + '\n',
                          encoding='utf-8')
