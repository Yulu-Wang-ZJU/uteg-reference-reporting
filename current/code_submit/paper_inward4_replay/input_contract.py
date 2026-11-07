"""Current source description and frozen analysis settings."""
import json
from pathlib import Path




def input_provenance():
    """Describe the supplied inputs and current runtime loading."""
    return {'source_status': 'hospital_supplied_inputs', 'source_basis': 'Participating hospitals.', 'source_statement': 'The participating hospitals supplied U-TEG MA values and voltage recordings, together with the corresponding instrument-exported TEG readouts.', 'runtime_input': "U-TEG inputs use a 900-point grid at one-second intervals; Supplementary Methods A1 defines the early windows. Within each window, processing masks identify retained samples, and any remaining positions are filled with that window's last retained value. Each prepared prefix is normalized by its own maximum absolute amplitude for fitting, with amplitude units restored afterwards. Parameter preparation precedes model evaluation. The runner loads saved coefficients and QC flags for each early window, and stored full-record coefficients at 15 minutes. Record-level QC determines input eligibility; the reporting gate separately determines release.", 'runtime_rescaling': False, 'runtime_reference_labels_used_to_build_features': False}




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
