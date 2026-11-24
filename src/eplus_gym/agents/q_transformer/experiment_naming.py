"""Stable experiment labels derived from active gates and auxiliary losses."""


def gate_aux_tag(config):
    parts = []
    if config.get('use_gate', False):
        for name in ('co2', 'iat'):
            weight = config.get(name + '_gate_weight', 0.0)
            if config.get('use_' + name + '_gate', False) and weight != 0:
                parts.append('{}_gate_{:g}'.format(name, weight))
    if not parts:
        parts.append('no_gate')
    auxiliary = []
    for name in ('co2', 'iat', 'occ'):
        weight = config.get(name + '_aux_weight', 0.0)
        if config.get('use_' + name + '_aux', False) and weight > 0:
            prefix = 'action_' if name in ('co2', 'iat') else ''
            target = 'occ_delta' if name == 'occ' and config.get('occ_predict_delta', False) else name
            auxiliary.append('{}{}_aux_{:g}'.format(prefix, target, weight))
    return '_'.join(parts + (auxiliary or ['no_aux']))
