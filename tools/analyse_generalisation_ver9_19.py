"""Compare the five figure metrics with equal seed and city weights."""
import csv
import itertools
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / 'runs' / 'metrics'
OUTPUT = ROOT / 'runs' / 'analysis' / 'ver9_19_seed101_103'
ARCHIVE = Path('D:/_/\u684c\u9762/\u5be6\u9a57\u5716/Generalization Test/Q-Transformer')
CITIES = ['Luxembourg', 'South Dakota', 'Perth', 'Brest', 'Lisbon']
SEEDS = (101, 102, 103)
FIELDS = ['average_score', 'indoor_violations', 'average_energy_mj',
          'total_co2_violations', 'total_iat_violations']
EXTRA = ['total_iat_high_violations', 'total_iat_low_violations']
PREFIX = 'eval_generalisation_q_transformer_summary_'
TAGS = {
    10: ('CO2', 'none', 'co2_iat_gated_aux_control_penalty_off'),
    11: ('none', 'none', 'baseline_control_penalty_off'),
    12: ('CO2', 'action_CO2', 'co2_gate_1_action_co2_aux_control_penalty_off'),
    13: ('CO2', 'action_IAT', 'co2_gate_1_action_iat_aux_control_penalty_off'),
    14: ('CO2', 'OCC', 'co2_gate_1_occ_aux_control_penalty_off'),
    15: ('CO2', 'action_CO2+action_IAT+OCC', 'co2_gate_1_action_co2_aux_action_iat_aux_occ_aux_control_penalty_off'),
    16: ('IAT', 'action_CO2', 'iat_gate_1_action_co2_aux_control_penalty_off'),
    17: ('IAT', 'action_IAT', 'iat_gate_1_action_iat_aux_control_penalty_off'),
    18: ('IAT', 'OCC', 'iat_gate_1_occ_aux_control_penalty_off'),
    19: ('IAT', 'action_CO2+action_IAT+OCC', 'iat_gate_1_action_co2_aux_action_iat_aux_occ_aux_control_penalty_off'),
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def docx_records(version):
    directory = next(p for p in ARCHIVE.iterdir() if p.name.startswith('Ver.{}_'.format(version)))
    path = directory / 'result.docx'
    with zipfile.ZipFile(path) as archive:
        tree = ET.fromstring(archive.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    text = '\n'.join(''.join(p.itertext()) for p in tree.findall('.//w:t', ns))
    labels = {
        'average_score': r'Average Score \(over 366 days\)',
        'indoor_violations': r'Total Indoor Violations \(occupant>0\)',
        'average_energy_mj': r'Average Daily Energy consumption \[MJ\]',
        'total_co2_violations': r'Total CO2 Violations \(CO2>1000, occupant>0\)',
        'total_iat_violations': r'Total IAT Violations \(occupant>0\)',
        'total_iat_high_violations': r'High IAT Violations \(IAT>24\)',
        'total_iat_low_violations': r'Low IAT Violations \(IAT<21\)',
    }
    records = []
    seed_matches = list(re.finditer(r'===== Running seed (\d+) =====', text))
    for index, seed_match in enumerate(seed_matches):
        seed = int(seed_match.group(1))
        section = text[seed_match.end():seed_matches[index + 1].start() if index + 1 < len(seed_matches) else len(text)]
        blocks = list(re.finditer(r'FINAL DEPLOYMENT STATS: (.*?) \(No Plots\)', section))
        for number, block in enumerate(blocks):
            body = section[block.end():blocks[number + 1].start() if number + 1 < len(blocks) else len(section)]
            row = dict(version=version, seed=seed, location=block.group(1), source=str(path))
            for field, label in labels.items():
                match = re.search(label + r':\s*([-+\d.eE]+)', body)
                if not match:
                    raise ValueError('Missing {} in {}'.format(field, row))
                row[field] = float(match.group(1))
            records.append(row)
    return records


def main():
    rows = docx_records(9)
    for version, (gate, aux, tag) in TAGS.items():
        for seed in SEEDS:
            path = METRICS / '{}{}_seed{}.csv'.format(PREFIX, tag, seed)
            for raw in read_csv(path):
                assert raw['control_penalty_enabled'].lower() == 'false'
                assert raw['iaq_aware_ventilation_enabled'].lower() == 'false'
                row = dict(version=version, seed=seed, location=raw['location'], source=str(path))
                for field in FIELDS + EXTRA:
                    row[field] = float(raw[field])
                    assert math.isfinite(row[field])
                rows.append(row)
    versions = [11, 10, 12, 13, 14, 15, 9, 16, 17, 18, 19]
    for version in versions:
        selected = [r for r in rows if r['version'] == version]
        assert len(selected) == 15
        assert {(r['seed'], r['location']) for r in selected} == set(itertools.product(SEEDS, CITIES))
    # Reconcile the original CSV/docx mappings against the archived version means.
    old_files = {
        9: 'co2_iat_gate_00_10_aux_control_penalty_off_seed_mean_from_docx',
        10: 'co2_iat_gate_10_00_no_aux_control_penalty_off_seed_mean_from_docx',
        11: 'baseline_control_penalty_off_seed_mean',
    }
    for version, tag in old_files.items():
        for expected in read_csv(METRICS / (PREFIX + tag + '.csv')):
            for field in FIELDS:
                actual = statistics.mean(r[field] for r in rows if r['version'] == version and r['location'] == expected['location'])
                assert abs(actual - float(expected[field])) < .0011, (version, field, actual, expected[field])
    city_means, seed_means, summary = [], [], []
    for version in versions:
        for city in CITIES:
            item = dict(version=version, location=city)
            for field in FIELDS + EXTRA:
                values = [r[field] for r in rows if r['version'] == version and r['location'] == city]
                item[field] = statistics.mean(values)
                item[field + '_seed_sd'] = statistics.stdev(values)
            city_means.append(item)
        for seed in SEEDS:
            item = dict(version=version, seed=seed)
            item.update({field: statistics.mean(r[field] for r in rows if r['version'] == version and r['seed'] == seed) for field in FIELDS + EXTRA})
            seed_means.append(item)
        item = dict(version=version, gate='IAT' if version == 9 else TAGS[version][0], aux='none' if version == 9 else TAGS[version][1])
        for field in FIELDS + EXTRA:
            values = [r[field] for r in seed_means if r['version'] == version]
            item[field] = statistics.mean(values)
            item[field + '_seed_sd'] = statistics.stdev(values)
        summary.append(item)
    def dominates(left, right):
        a = [-left['average_score']] + [left[f] for f in FIELDS[1:]]
        b = [-right['average_score']] + [right[f] for f in FIELDS[1:]]
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
    for item in summary:
        item['dominated_by'] = ','.join(str(other['version']) for other in summary if dominates(other, item))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, records in [('source_records.csv', rows), ('city_seed_means.csv', city_means),
                              ('seed_five_city_means.csv', seed_means), ('overall_comparison.csv', summary)]:
        with (OUTPUT / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print('OVERALL: version gate aux | score, indoor, energy, CO2, IAT | seed SD of five-city means')
    for r in summary:
        print('V{} {} {} | {} | SD {} | dominated by {}'.format(r['version'], r['gate'], r['aux'],
            ', '.join('{:.3f}'.format(r[f]) for f in FIELDS),
            ', '.join('{:.3f}'.format(r[f + '_seed_sd']) for f in FIELDS), r['dominated_by']))
    lookup = {r['version']: r for r in summary}
    print('\nPERCENT CHANGE vs baseline (score absolute point change):')
    base = lookup[11]
    for r in summary:
        print('V{}: score {:+.3f}; {}'.format(r['version'], r['average_score']-base['average_score'],
            ', '.join('{} {:+.2f}%'.format(f, 100*(r[f]/base[f]-1)) for f in FIELDS[1:])))
    print('\nAUX EFFECT vs same-gate no-aux (score absolute point change):')
    for control, candidates in [(10, [12, 13, 14, 15]), (9, [16, 17, 18, 19])]:
        base = lookup[control]
        for version in candidates:
            r = lookup[version]
            print('V{} vs V{}: score {:+.3f}; {}'.format(version, control, r['average_score']-base['average_score'],
                ', '.join('{} {:+.2f}%'.format(f, 100*(r[f]/base[f]-1)) for f in FIELDS[1:])))
    print('\nCITY MEANS:')
    for r in city_means:
        print('V{} {}: {}'.format(r['version'], r['location'], ', '.join('{} {:.3f}'.format(f, r[f]) for f in FIELDS)))
    print('\nSEED MEANS:')
    for r in seed_means:
        print('V{} seed {}: {}'.format(r['version'], r['seed'], ', '.join('{:.3f}'.format(r[f]) for f in FIELDS)))
    print('Saved:', OUTPUT)


if __name__ == '__main__':
    main()
