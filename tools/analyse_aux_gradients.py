"""Summarize training diagnostic CSVs by seed and episode without loading EnergyPlus."""
import argparse
import csv
import glob
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(paths):
    groups = defaultdict(list)
    for path in paths:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                groups[(row['seed'], 'all')].append(row)
                groups[(row['seed'], row['episode'])].append(row)
    results = []
    for (seed, episode), rows in groups.items():
        cosine = [float(r['q_aux_grad_cosine']) for r in rows
                  if r['gradient_direction_valid'] == 'True']
        ratio = [float(r['aux_to_q_grad_ratio']) for r in rows
                 if math.isfinite(float(r['aux_to_q_grad_ratio']))]
        item = dict(seed=seed, episode=episode, updates=len(rows),
                    valid_directions=len(cosine), undefined_directions=len(rows) - len(cosine),
                    conflict_percent=100 * sum(c < 0 for c in cosine) / len(cosine) if cosine else float('nan'),
                    cosine_mean=statistics.mean(cosine) if cosine else float('nan'),
                    cosine_median=statistics.median(cosine) if cosine else float('nan'),
                    ratio_mean=statistics.mean(ratio) if ratio else float('nan'),
                    ratio_median=statistics.median(ratio) if ratio else float('nan'),
                    ratio_max=max(ratio) if ratio else float('nan'))
        for name in ('q_loss', 'occ_aux_loss', 'weighted_aux_loss', 'total_loss',
                     'q_encoder_grad_norm', 'occ_encoder_grad_norm', 'weighted_aux_encoder_grad_norm'):
            values = [float(r[name]) for r in rows if math.isfinite(float(r[name]))]
            item[name + '_mean'] = statistics.mean(values) if values else float('nan')
        # Sum squared errors and counts: batches contain different numbers of occupancy changes.
        available = [r for r in rows if r.get('occ_sample_count')]
        for prefix, count_field in (('occ', 'occ_sample_count'), ('occ_changed', 'occ_changed_count')):
            count = sum(int(r[count_field]) for r in available)
            item[count_field] = count
            for method in ('prediction', 'persistence'):
                sse = sum(float(r[prefix + '_' + method + '_sse']) for r in available)
                item[prefix + '_' + method + '_mse'] = sse / count if count else float('nan')
            base = item[prefix + '_persistence_mse']
            item[prefix + '_mse_improvement_pct'] = (
                100 * (1 - item[prefix + '_prediction_mse'] / base) if base > 0 else float('nan')
            )
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', help='Diagnostic CSV paths or quoted glob patterns.')
    parser.add_argument('--output', type=Path, help='Optional per-seed and per-episode summary CSV.')
    args = parser.parse_args()
    paths = sorted({Path(p) for pattern in args.files for p in glob.glob(pattern)})
    if not paths:
        parser.error('No input files matched.')
    if args.output and args.output.resolve() in {p.resolve() for p in paths}:
        parser.error('Output must not overwrite an input file.')
    results = summarize(paths)
    for row in results:
        if row['episode'] == 'all':
            print('Seed {}: {} updates, conflict {:.2f}% of {} valid directions, '
                  'mean cosine {:.4f}, median ratio {:.6g}, mean ratio {:.6g}'.format(
                      row['seed'], row['updates'], row['conflict_percent'], row['valid_directions'],
                      row['cosine_mean'], row['ratio_median'], row['ratio_mean']))
            if row['occ_sample_count']:
                print('  OCC MSE: model {:.6g}, persistence {:.6g}, improvement {:.2f}%; '
                      'changed samples {}: model {:.6g}, persistence {:.6g}, improvement {:.2f}%'.format(
                          row['occ_prediction_mse'], row['occ_persistence_mse'], row['occ_mse_improvement_pct'],
                          row['occ_changed_count'], row['occ_changed_prediction_mse'],
                          row['occ_changed_persistence_mse'], row['occ_changed_mse_improvement_pct']))
    if args.output and results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print('Saved:', args.output)


if __name__ == '__main__':
    main()
