"""Analyze the new training run only; do not reuse older evaluation CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyse_ver20 import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'analysis' / 'occ_0.1_seed101'
PATH = ROOT / 'runs' / 'metrics' / 'generalisation_Q-transformer_co2_gate_1_occ_aux_0.1_gradient_diagnostics_seed101.csv'


def add_prediction_summary(frame, row):
    for prefix, count_field in [('occ', 'occ_sample_count'), ('occ_changed', 'occ_changed_count')]:
        count = frame[count_field].sum()
        row[count_field] = count
        for method in ['prediction', 'persistence']:
            row[prefix+'_'+method+'_mse'] = frame[prefix+'_'+method+'_sse'].sum()/count if count else np.nan
        row[prefix+'_model_to_persistence'] = row[prefix+'_prediction_mse']/row[prefix+'_persistence_mse']
    unchanged_count = row['occ_sample_count'] - row['occ_changed_count']
    row['occ_unchanged_prediction_mse'] = (
        (frame.occ_prediction_sse.sum()-frame.occ_changed_prediction_sse.sum())/unchanged_count)
    row['changed_sample_pct'] = 100*row['occ_changed_count']/row['occ_sample_count']
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PATH)
    assert df.seed.eq(101).all() and df.occ_aux_weight.eq(.1).all()
    assert df.co2_aux_weight.eq(0).all() and df.iat_aux_weight.eq(0).all()
    assert df.co2_gate_weight.eq(1).all() and df.iat_gate_weight.eq(0).all()
    assert np.array_equal(df.learning_step.values, np.arange(1, len(df)+1))
    assert df.gradient_direction_valid.all()
    assert np.allclose(df.aux_to_q_grad_ratio, df.weighted_aux_encoder_grad_norm/df.q_encoder_grad_norm)
    assert np.allclose(df.weighted_aux_encoder_grad_norm, .1*df.occ_encoder_grad_norm)
    assert np.allclose(df.occ_prediction_sse, df.occ_prediction_mse*df.occ_sample_count)
    assert np.allclose(df.occ_prediction_mse, df.occ_aux_loss)
    assert np.allclose(df.weighted_aux_loss, .1*df.occ_aux_loss)
    phases = [('all', df), ('ep1_30', df[df.episode <= 30]),
              ('ep31_90', df[(df.episode > 30)&(df.episode <= 90)]),
              ('ep91_183', df[df.episode > 90]), ('last30', df[df.episode > 153])]
    rows = [add_prediction_summary(part, summarize(part, 101, name)) for name, part in phases]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/'phase_summary.csv', index=False)
    episodes = pd.DataFrame([add_prediction_summary(part, summarize(part, 101, str(ep)))
                             for ep, part in df.groupby('episode')])
    episodes.to_csv(OUT/'episode_summary.csv', index=False)
    print('PHASES:')
    columns = ['phase', 'updates', 'conflict_pct', 'cosine_mean', 'ratio_median', 'ratio_p95',
               'ratio_max', 'ratio_gt_0p1_pct', 'opposing_q_fraction_mean',
               'opposing_gt_0p01_pct', 'opposing_gt_0p1_pct', 'combined_gradient_reverses_q_pct',
               'q_loss_mean', 'occ_prediction_mse', 'occ_persistence_mse', 'occ_model_to_persistence',
               'occ_changed_prediction_mse', 'occ_changed_persistence_mse',
               'occ_changed_model_to_persistence', 'occ_unchanged_prediction_mse', 'changed_sample_pct']
    print(summary[columns].to_string(index=False))
    old = pd.read_csv(ROOT/'runs/analysis/ver20_occ_0.01/gradient_phase_summary.csv')
    old = old[(old.seed == 101)&(old.phase == 'all')].iloc[0]
    new = summary[summary.phase == 'all'].iloc[0]
    print('Weight 0.1 vs 0.01, same seed:')
    for field in ['conflict_pct','cosine_mean','ratio_median','ratio_mean','q_loss_mean','occ_aux_loss_mean']:
        print(field, old[field], '->', new[field])
    print('Total replay sample draws:', df.occ_sample_count.sum(), 'Changed:', df.occ_changed_count.sum())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0,0].plot(episodes.phase.astype(int), episodes.occ_prediction_mse, label='OCC model')
    axes[0,0].plot(episodes.phase.astype(int), episodes.occ_persistence_mse, label='Persistence')
    axes[0,0].set_title('All replay samples: normalized MSE')
    axes[0,1].plot(episodes.phase.astype(int), episodes.occ_changed_prediction_mse, label='OCC model')
    axes[0,1].plot(episodes.phase.astype(int), episodes.occ_changed_persistence_mse, label='Persistence')
    axes[0,1].set_title('Occupancy-change samples: normalized MSE')
    axes[1,0].plot(episodes.phase.astype(int), episodes.ratio_median, label='Weighted OCC / Q norm (median)')
    axes[1,0].set_title('Encoder gradient strength')
    axes[1,0].set_yscale('log')
    axes[1,1].plot(episodes.phase.astype(int), episodes.conflict_pct, label='Negative cosine [%]')
    axes[1,1].set_title('Encoder gradient conflict rate')
    for ax in axes.ravel():
        ax.set_xlabel('Episode')
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'training_diagnostics.png', dpi=160)
    plt.close(fig)
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
