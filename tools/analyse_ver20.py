"""Audit Ver.20 control results and encoder-gradient diagnostics."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / 'runs' / 'metrics'
OUT = ROOT / 'runs' / 'analysis' / 'ver20_occ_0.01'
FIELDS = ['average_score', 'indoor_violations', 'average_energy_mj',
          'total_co2_violations', 'total_iat_violations']
EXTRA = ['total_iat_high_violations', 'total_iat_low_violations']
SEEDS = [101, 102, 103]
CITIES = ['Luxembourg', 'South Dakota', 'Perth', 'Brest', 'Lisbon']


def summarize(frame, seed, phase):
    valid = frame['gradient_direction_valid'] & np.isfinite(frame['q_aux_grad_cosine'])
    data = frame.loc[valid]
    ratio = data['aux_to_q_grad_ratio']
    cosine = data['q_aux_grad_cosine']
    opposing = (-ratio * cosine).clip(lower=0)
    row = dict(seed=seed, phase=phase, updates=len(frame), valid=len(data),
               conflict_pct=100 * (cosine < 0).mean(), cosine_mean=cosine.mean(),
               cosine_median=cosine.median(), cosine_p05=cosine.quantile(.05),
               cosine_p95=cosine.quantile(.95), cosine_lt_minus_0p1_pct=100*(cosine < -.1).mean(),
               cosine_lt_minus_0p5_pct=100*(cosine < -.5).mean(),
               ratio_mean=ratio.mean(), ratio_median=ratio.median(), ratio_p95=ratio.quantile(.95),
               ratio_p99=ratio.quantile(.99), ratio_max=ratio.max(),
               ratio_gt_0p01_pct=100*(ratio > .01).mean(), ratio_gt_0p1_pct=100*(ratio > .1).mean(),
               ratio_gt_1_pct=100*(ratio > 1).mean(),
               opposing_q_fraction_mean=opposing.mean(), opposing_q_fraction_max=opposing.max(),
               opposing_gt_0p01_pct=100*(opposing > .01).mean(),
               opposing_gt_0p1_pct=100*(opposing > .1).mean(),
               combined_gradient_reverses_q_pct=100*(1 + ratio*cosine < 0).mean())
    for field in ['q_loss', 'occ_aux_loss', 'weighted_aux_loss', 'total_loss',
                  'q_encoder_grad_norm', 'occ_encoder_grad_norm', 'weighted_aux_encoder_grad_norm']:
        row[field+'_mean'] = frame[field].mean()
        row[field+'_median'] = frame[field].median()
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    tags = {
        10: 'co2_iat_gated_aux_control_penalty_off',
        11: 'baseline_control_penalty_off',
        14: 'co2_gate_1_occ_aux_0.1_control_penalty_off',
        20: 'co2_gate_1_occ_aux_0.01_control_penalty_off',
    }
    for version, tag in tags.items():
        for seed in SEEDS:
            path = METRICS / ('eval_generalisation_q_transformer_summary_'+tag+'_seed'+str(seed)+'.csv')
            df = pd.read_csv(path)
            assert len(df) == 5 and set(df.location) == set(CITIES)
            assert not df.control_penalty_enabled.any() and not df.iaq_aware_ventilation_enabled.any()
            df['version'], df['seed'], df['source'] = version, seed, str(path)
            all_rows.append(df)
    raw = pd.concat(all_rows, ignore_index=True)
    city = raw.groupby(['version', 'location'])[FIELDS + EXTRA].mean().reset_index()
    seed_mean = raw.groupby(['version', 'seed'])[FIELDS + EXTRA].mean().reset_index()
    overall = seed_mean.groupby('version')[FIELDS + EXTRA].agg(['mean', 'std'])
    overall.columns = ['_'.join(c) for c in overall.columns]
    overall = overall.reset_index()
    city.to_csv(OUT / 'city_means.csv', index=False)
    seed_mean.to_csv(OUT / 'seed_five_city_means.csv', index=False)
    overall.to_csv(OUT / 'control_comparison.csv', index=False)
    print('CONTROL COMPARISON:')
    print(overall.to_string(index=False))
    print('V20 seed means:')
    print(seed_mean[seed_mean.version == 20].to_string(index=False))
    print('V20 city means:')
    print(city[city.version == 20].to_string(index=False))
    for version in [10, 11, 14]:
        old = overall.set_index('version').loc[version]
        new = overall.set_index('version').loc[20]
        print('V20 vs V{}: score delta {:.5f}; {}'.format(version,
              new.average_score_mean-old.average_score_mean,
              ', '.join('{} {:+.3f}%'.format(f, 100*(new[f+'_mean']/old[f+'_mean']-1)) for f in FIELDS[1:])))
    summaries, episode_frames = [], []
    for seed in SEEDS:
        path = METRICS / ('generalisation_Q-transformer_co2_gate_1_occ_aux_0.01_gradient_diagnostics_seed'+str(seed)+'.csv')
        df = pd.read_csv(path)
        assert df.seed.eq(seed).all() and df.occ_aux_weight.eq(.01).all()
        assert df.co2_aux_weight.eq(0).all() and df.iat_aux_weight.eq(0).all()
        assert df.co2_gate_weight.eq(1).all() and df.iat_gate_weight.eq(0).all()
        assert np.array_equal(df.learning_step.values, np.arange(1, len(df)+1))
        assert df.environment_step.is_monotonic_increasing
        assert df.episode.max() == 183
        numeric = df.select_dtypes(include=[np.number])
        print('Seed {}: updates {}, nonfinite entries {}, first/last environment step {}/{}, invalid directions {}'.format(
              seed, len(df), (~np.isfinite(numeric)).sum().sum(), df.environment_step.iloc[0],
              df.environment_step.iloc[-1], (~df.gradient_direction_valid).sum()))
        valid = df.gradient_direction_valid
        assert np.allclose(df.loc[valid, 'aux_to_q_grad_ratio'],
                           df.loc[valid, 'weighted_aux_encoder_grad_norm']/df.loc[valid, 'q_encoder_grad_norm'])
        assert np.allclose(df.weighted_aux_encoder_grad_norm, .01*df.occ_encoder_grad_norm)
        assert np.allclose(df.weighted_aux_loss, .01*df.occ_aux_loss)
        assert np.allclose(df.total_loss, df.q_loss+df.weighted_aux_loss)
        assert (df.loc[valid, 'gradient_conflict'] == (df.loc[valid, 'q_aux_grad_cosine'] < 0)).all()
        phases = [('all', df), ('ep1_30', df[df.episode <= 30]),
                  ('ep31_90', df[(df.episode > 30) & (df.episode <= 90)]),
                  ('ep91_183', df[df.episode > 90]), ('last30', df[df.episode > 153])]
        summaries.extend(summarize(part, seed, name) for name, part in phases)
        for episode, part in df.groupby('episode'):
            row = summarize(part, seed, str(episode))
            row['episode'] = episode
            episode_frames.append(row)
    diagnostics = pd.DataFrame(summaries)
    episodes = pd.DataFrame(episode_frames)
    diagnostics.to_csv(OUT/'gradient_phase_summary.csv', index=False)
    episodes.to_csv(OUT/'gradient_episode_summary.csv', index=False)
    print('GRADIENT PHASES:')
    print(diagnostics.to_string(index=False))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = [('conflict_pct', 'Negative cosine updates [%]', False),
             ('cosine_mean', 'Mean cosine(Q, OCC)', False),
             ('ratio_median', 'Median ||0.01 g_OCC|| / ||g_Q||', True),
             ('occ_aux_loss_mean', 'Mean OCC training MSE (normalized)', True)]
    for ax, (field, title, log) in zip(axes.ravel(), specs):
        for seed in SEEDS:
            data = episodes[episodes.seed == seed]
            ax.plot(data.episode, data[field], label='Seed '+str(seed), linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        if log:
            ax.set_yscale('log')
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'gradient_training_trends.png', dpi=160)
    plt.close(fig)
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
