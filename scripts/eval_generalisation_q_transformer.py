# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 07:18:07 2026

@author: khali
"""
#%% library part

# Standard library
import os
import pickle
from collections import deque
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local project modules
from eplus_gym.envs.env import AmphitheaterEnv
from eplus_gym.envs.energyplus import (
    _resolve_output_dir,
    _find_project_root,
)
from eplus_gym.agents.q_transformer.dqn_agent import Q_transformer


# -----------------------------------------------------------------------------
# Helper utilities                                                            |
# -----------------------------------------------------------------------------

def _ensure_dir(path: str):
    """Create parent directory if it does not yet exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_training_metrics(filepath: str, scores, eps_history, steps_array):
    """Persist training arrays to *filepath* (Pickle)."""
    _ensure_dir(filepath)
    with open(filepath, "wb") as f:
        pickle.dump({
            "scores": scores,
            "eps_history": eps_history,
            "steps_array": steps_array,
        }, f)


def _count_short_cycles(binary_series, min_stable_steps: int = 4) -> int:
    """Count AHU state segments shorter than the target stable duration."""
    arr = np.asarray(binary_series, dtype=int)
    if arr.size == 0:
        return 0

    switch_points = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    segment_starts = np.r_[0, switch_points]
    segment_ends = np.r_[switch_points, arr.size]
    durations = segment_ends - segment_starts
    return int(np.sum(durations < min_stable_steps))


def _control_variation_metrics(series, valid_transition_mask):
    """Return total, mean, and max absolute setpoint changes for valid transitions."""
    arr = np.asarray(series, dtype=float)
    valid = np.asarray(valid_transition_mask, dtype=bool)
    diffs = np.abs(np.diff(arr))
    selected = diffs[valid]

    if selected.size == 0:
        return 0.0, 0.0, 0.0

    return (
        float(np.sum(selected)),
        float(np.mean(selected)),
        float(np.max(selected)),
    )


output_dir = Path(_resolve_output_dir(None))
output_root = Path(_resolve_output_dir(None))
(output_root / ".gitkeep").touch(exist_ok=True)

checkpoint_dir = _resolve_output_dir(None, default_name="runs/checkpoints")

ALGO_NAME = "CO2-gated-aux-Q-transformer"
EXPERIMENT_TAG = "co2_gated_aux"

CONTROL_PENALTY_CONFIG = {
    "enable_control_penalty": False,
    "w_switch": 0.3,
    "w_du_sat": 0.04,
    "w_du_oac": 0.04,
    "enable_iaq_aware_ventilation": False,
    "iaq_co2_threshold": 850.0,
    "iaq_switch_penalty_scale": 0.5,
    "iaq_oac_penalty_scale": 0.35,
    "iaq_oac_energy_scale": 0.4,
    "iaq_ahu_energy_scale": 0.35,
    "iaq_target_oac_fraction": 0.4,
    "iaq_high_co2_threshold": 950.0,
    "iaq_high_target_oac_fraction": 0.65,
    "w_iaq_ventilation_deficit": 1.0,
    "iaq_co2_severity_threshold": 1000.0,
    "w_iaq_co2_severity": 0.8,
}


#%% Generalisation test (Q-transformer)

n_days = 366
SHORT_CYCLE_MIN_STEPS = 4

EVAL_LOCATIONS = [
    ("Luxembourg", "luxembourg"),
    ("South Dakota", "usa"),
    ("Perth", "australia"),
    ("Brest", "brest"),
    ("Lisbon", "portugal"),
]


def evaluate_location(location_label: str, location_key: str):
    env = AmphitheaterEnv(
        env_config={"output": str(output_dir), "location": location_key, **CONTROL_PENALTY_CONFIG},
        new_begin_month=1,
        new_end_month=12,
        new_begin_day=1,
        new_end_day=31,
        train=False,
    )
    env.runner_config.csv = True

    agent = Q_transformer(
        gamma=0.99,
        epsilon=0,
        lr=0.001,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=40000,
        eps_min=0.05,
        batch_size=128,
        replace=384,
        eps_dec=0.9 * n_days * 366,
        chkpt_dir=checkpoint_dir,
        algo=ALGO_NAME,
        env_name='energyplus'
    )

    agent.load_models()
    agent.q_eval.eval()

    scores = []
    action_temp = []
    action_flow = []
    action_proxy_cost = []
    ahu_cmd_series = []

    state_window = deque(maxlen=96)
    done = False
    observation = env.reset()[0]
    score = 0

    while not done:
        if len(state_window) < 96:
            action_idx = np.random.choice(agent.action_space)
        else:
            obs_seq = np.array(state_window, dtype=np.float32)
            action_idx = agent.choose_action_test(obs_seq)

        observation_, reward, terminated, truncated, _ = env.step(action_idx)
        action = env.valid_actions[action_idx]

        score += reward
        done = terminated or truncated
        observation = observation_
        state_window.append(observation)

        ahu_cmd_series.append(int(action[2]))
        action_proxy_cost.append(
            (action[0] / 7) * action[2]
            + (action[1] / 7) * action[2]
            + action[2] / 5
        )
        action_temp.append(
            env._rescale(action[0] * action[2], range1=(0, 8-1), range2=[15, 30])
        )
        action_flow.append(
            env._rescale(action[1] * action[2], range1=(0, 8-1), range2=[0.3, 5])
        )
        scores.append(reward)

    print(f"Score (entire run, {location_label}): ", score)
    env.close()

    total_score = 0.0
    co2_violation_count = 0
    iat_violation_high_count = 0
    iat_violation_low_count = 0
    Indoor_violation_count = 0
    Heating_Distict_total = 0.0
    Electricity_Hvac_total = 0.0
    Electricity_Plant_total = 0.0
    Heating_Coil_total = 0.0
    CO2_Concentration_total = 0.0

    first_timestep = -96
    last_timestep = 0

    csv_path = (_find_project_root() / "eplus_outputs" / "eplusout.csv")
    data = pd.read_csv(csv_path)

    data['Date/Time'] = '2020/' + data['Date/Time'].str.strip()
    data['Date/Time'] = data['Date/Time'].str.replace(r'\s+', ' ', regex=True)
    data['Date/Time'] = data['Date/Time'].str.replace('24:00:00', '23:59:59')
    data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%Y/%m/%d %H:%M:%S')

    analysis_len = min(len(scores), len(data), len(action_temp), len(action_flow), len(ahu_cmd_series))
    analysis_data = data.head(analysis_len).copy()
    occupied_mask = (
        analysis_data['MAISON DU SAVOIR AUDITORIUM OCC:Schedule Value [](TimeStep)']
        .to_numpy(dtype=float)
        > 0
    )
    occupied_transition_mask = occupied_mask[1:] & occupied_mask[:-1]

    ahu_cmd_array = np.asarray(ahu_cmd_series[:analysis_len], dtype=int)
    n_switch_cmd = int(np.sum(ahu_cmd_array[1:] != ahu_cmd_array[:-1])) if ahu_cmd_array.size > 1 else 0

    ahu_actual_array = analysis_data['AHUS ONOFF:Schedule Value [](TimeStep)'].to_numpy(dtype=int)
    n_switch_actual = int(np.sum(ahu_actual_array[1:] != ahu_actual_array[:-1])) if ahu_actual_array.size > 1 else 0
    short_cycle_count = _count_short_cycles(
        ahu_actual_array,
        min_stable_steps=SHORT_CYCLE_MIN_STEPS,
    )

    action_proxy_energy_cost = float(np.sum(action_proxy_cost[:analysis_len]))
    total_electricity_hvac_j = float(analysis_data['Electricity:HVAC [J](TimeStep)'].sum())
    total_district_heating_j = float(analysis_data['Heating:DistrictHeatingWater [J](TimeStep)'].sum())

    zone_t = analysis_data['TZ_AMPHITHEATER:Zone Mean Air Temperature [C](TimeStep)'].to_numpy(dtype=float)
    htg_thr = analysis_data['HTG HVAC 1 ADJUSTED BY 1.1 F:Schedule Value [](TimeStep)'].to_numpy(dtype=float)
    clg_thr = analysis_data['CLG HVAC 1 ADJUSTED BY 0 F:Schedule Value [](TimeStep)'].to_numpy(dtype=float)
    co2_values = analysis_data['TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)'].to_numpy(dtype=float)
    occupied_temp_violation = int(np.sum(occupied_mask & ((zone_t < htg_thr) | (zone_t > clg_thr))))
    occupied_co2_violation = int(np.sum(occupied_mask & (co2_values > 1000.0)))

    (
        sat_total_variation_occ,
        sat_avg_abs_change_occ,
        sat_max_jump_occ,
    ) = _control_variation_metrics(action_temp[:analysis_len], occupied_transition_mask)
    (
        oa_total_variation_occ,
        oa_avg_abs_change_occ,
        oa_max_jump_occ,
    ) = _control_variation_metrics(action_flow[:analysis_len], occupied_transition_mask)

    for _ in range(n_days):
        first_timestep += 96
        last_timestep += 96

        chunk_score = float(
            pd.DataFrame(scores)
            .head(last_timestep)
            .tail(last_timestep - first_timestep)
            .sum()
            .iloc[0]
        )
        total_score += chunk_score

        chunk_data = data.head(last_timestep).tail(last_timestep - first_timestep).copy()

        temp_zone = chunk_data['TZ_AMPHITHEATER:Zone Mean Air Temperature [C](TimeStep)']
        occupancy = chunk_data['MAISON DU SAVOIR AUDITORIUM OCC:Schedule Value [](TimeStep)']
        co2_conc = chunk_data['TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)']

        Indoor_violation_chunk = (
            ((co2_conc > 1000) | (temp_zone > 24) | (temp_zone < 21)) & (occupancy > 0)
        ).sum()
        Indoor_violation_count += Indoor_violation_chunk

        co2_violation_chunk = ((co2_conc > 1000) & (occupancy > 0)).sum()
        co2_violation_count += co2_violation_chunk

        iat_high_chunk = ((temp_zone > 24) & (occupancy > 0)).sum()
        iat_low_chunk = ((temp_zone < 21) & (occupancy > 0)).sum()
        iat_violation_high_count += iat_high_chunk
        iat_violation_low_count += iat_low_chunk

        electricity_hvac = chunk_data['Electricity:HVAC [J](TimeStep)']
        heating_district = chunk_data['Heating:DistrictHeatingWater [J](TimeStep)']
        electricity_plant = chunk_data['Electricity:Plant [J](TimeStep)']
        heating_coil = chunk_data['HeatingCoils:EnergyTransfer [J](TimeStep)']

        Heating_Distict_total += heating_district.sum()
        Electricity_Hvac_total += electricity_hvac.sum()
        Electricity_Plant_total += electricity_plant.sum()
        Heating_Coil_total += heating_coil.sum()
        CO2_Concentration_total += co2_conc.sum()

    average_score = total_score / n_days
    iat_violation_count = iat_violation_high_count + iat_violation_low_count

    avg_heating_district_MJ = (Heating_Distict_total / n_days) / 1e6
    avg_elec_hvac_MJ = (Electricity_Hvac_total / n_days) / 1e6
    avg_elec_plant_MJ = (Electricity_Plant_total / n_days) / 1e6
    avg_heating_coil_MJ = (Heating_Coil_total / n_days) / 1e6
    avg_ene = avg_heating_district_MJ + avg_elec_hvac_MJ

    print(f"\n============ FINAL DEPLOYMENT STATS: {location_label} (No Plots) ============")
    print(f"Average Score (over {n_days} days): {average_score:.3f}")
    print(f"Total Indoor Violations (occupant>0): {Indoor_violation_count}")
    print(f"Total CO2 Violations (CO2>1000, occupant>0): {co2_violation_count}")
    print(f"Total IAT Violations (occupant>0): {iat_violation_count}")
    print(f"  --> High IAT Violations (IAT>24): {iat_violation_high_count}")
    print(f"  --> Low IAT Violations (IAT<21): {iat_violation_low_count}")
    print(f"Average Daily Energy consumption [MJ]: {avg_ene:.3f}")
    print(f"Average Daily District Heating [MJ]: {avg_heating_district_MJ:.3f}")
    print(f"Average Daily HVAC Electricity [MJ]: {avg_elec_hvac_MJ:.3f}")
    print(f"Average Daily Plant Electricity [MJ]: {avg_elec_plant_MJ:.3f}")
    print(f"Average Daily Heating Coil [MJ]: {avg_heating_coil_MJ:.3f}")
    print("AHU switching count (command):", n_switch_cmd)
    print("AHU switching count (actual state):", n_switch_actual)
    print("occupied_temp_violation:", occupied_temp_violation)
    print("occupied_co2_violation:", occupied_co2_violation)
    print("action_proxy_energy_cost:", action_proxy_energy_cost)
    print("total_electricity_hvac [J]:", total_electricity_hvac_j)
    print("total_district_heating [J]:", total_district_heating_j)
    print(f"short_cycle_count (<{SHORT_CYCLE_MIN_STEPS} timesteps, actual AHU):", short_cycle_count)
    print("SAT total variation (occupied only):", sat_total_variation_occ)
    print("SAT average absolute change (occupied only):", sat_avg_abs_change_occ)
    print("SAT maximum jump (occupied only):", sat_max_jump_occ)
    print("OA flow total variation (occupied only):", oa_total_variation_occ)
    print("OA flow average absolute change (occupied only):", oa_avg_abs_change_occ)
    print("OA flow maximum jump (occupied only):", oa_max_jump_occ)
    print("===========================================================\n")

    return {
        "location": location_label,
        "average_score": average_score,
        "indoor_violations": Indoor_violation_count,
        "total_co2_violations": co2_violation_count,
        "total_iat_violations": iat_violation_count,
        "total_iat_high_violations": iat_violation_high_count,
        "total_iat_low_violations": iat_violation_low_count,
        "average_energy_mj": avg_ene,
        "ahu_switching_count_command": n_switch_cmd,
        "ahu_switching_count_actual": n_switch_actual,
        "occupied_temp_violation": occupied_temp_violation,
        "occupied_co2_violation": occupied_co2_violation,
        "action_proxy_energy_cost": action_proxy_energy_cost,
        "total_electricity_hvac_j": total_electricity_hvac_j,
        "total_district_heating_j": total_district_heating_j,
        "short_cycle_count": short_cycle_count,
        "sat_total_variation_occupied": sat_total_variation_occ,
        "sat_avg_abs_change_occupied": sat_avg_abs_change_occ,
        "sat_max_jump_occupied": sat_max_jump_occ,
        "oa_flow_total_variation_occupied": oa_total_variation_occ,
        "oa_flow_avg_abs_change_occupied": oa_avg_abs_change_occ,
        "oa_flow_max_jump_occupied": oa_max_jump_occ,
        "control_penalty_enabled": CONTROL_PENALTY_CONFIG["enable_control_penalty"],
        "w_switch": CONTROL_PENALTY_CONFIG["w_switch"],
        "w_du_sat": CONTROL_PENALTY_CONFIG["w_du_sat"],
        "w_du_oac": CONTROL_PENALTY_CONFIG["w_du_oac"],
        "iaq_aware_ventilation_enabled": CONTROL_PENALTY_CONFIG["enable_iaq_aware_ventilation"],
        "iaq_co2_threshold": CONTROL_PENALTY_CONFIG["iaq_co2_threshold"],
        "iaq_switch_penalty_scale": CONTROL_PENALTY_CONFIG["iaq_switch_penalty_scale"],
        "iaq_oac_penalty_scale": CONTROL_PENALTY_CONFIG["iaq_oac_penalty_scale"],
        "iaq_oac_energy_scale": CONTROL_PENALTY_CONFIG["iaq_oac_energy_scale"],
        "iaq_ahu_energy_scale": CONTROL_PENALTY_CONFIG["iaq_ahu_energy_scale"],
        "iaq_target_oac_fraction": CONTROL_PENALTY_CONFIG["iaq_target_oac_fraction"],
        "iaq_high_co2_threshold": CONTROL_PENALTY_CONFIG["iaq_high_co2_threshold"],
        "iaq_high_target_oac_fraction": CONTROL_PENALTY_CONFIG["iaq_high_target_oac_fraction"],
        "w_iaq_ventilation_deficit": CONTROL_PENALTY_CONFIG["w_iaq_ventilation_deficit"],
        "iaq_co2_severity_threshold": CONTROL_PENALTY_CONFIG["iaq_co2_severity_threshold"],
        "w_iaq_co2_severity": CONTROL_PENALTY_CONFIG["w_iaq_co2_severity"],
    }


def _annotate_bars(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        y_offset = 3 if height >= 0 else -14
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )


def save_summary_plot(results):
    plots_dir = Path(_resolve_output_dir(None, default_name="runs/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "eval_generalisation_co2_gated_aux_q_transformer_summary.png"

    locations = [item["location"] for item in results]
    average_scores = [item["average_score"] for item in results]
    indoor_violations = [item["indoor_violations"] for item in results]
    average_energy = [item["average_energy_mj"] for item in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    color = "tab:orange"

    bars = axes[0].bar(locations, average_scores, color=color, label="Q-transformer")
    axes[0].set_title("(c) Average Score")
    axes[0].set_ylabel("Average score")
    axes[0].legend()
    _annotate_bars(axes[0], bars, "{:.2f}")

    bars = axes[1].bar(locations, indoor_violations, color=color, label="Q-transformer")
    axes[1].set_title("(d) Indoor Violation (occupied periods)")
    axes[1].set_ylabel("Indoor violation")
    axes[1].legend()
    _annotate_bars(axes[1], bars, "{:.0f}")

    bars = axes[2].bar(locations, average_energy, color=color, label="Q-transformer")
    axes[2].set_title("(e) Average Energy Consumption [MJ]")
    axes[2].set_ylabel("Average energy consumption")
    axes[2].legend()
    _annotate_bars(axes[2], bars, "{:.3f}")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved plot] {plot_path}")


def save_summary_metrics(results):
    metrics_dir = Path(_resolve_output_dir(None, default_name="runs/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tag_parts = [
        EXPERIMENT_TAG,
        "control_penalty_on" if CONTROL_PENALTY_CONFIG["enable_control_penalty"] else "control_penalty_off",
    ]
    if CONTROL_PENALTY_CONFIG.get("enable_iaq_aware_ventilation", False):
        tag_parts.append("iaq_on")
    tag = "_".join(tag_parts)
    df = pd.DataFrame(results)
    metrics_path = metrics_dir / "eval_generalisation_q_transformer_summary.csv"
    tagged_metrics_path = metrics_dir / f"eval_generalisation_q_transformer_summary_{tag}.csv"
    df.to_csv(metrics_path, index=False)
    df.to_csv(tagged_metrics_path, index=False)
    print(f"[Saved metrics] {metrics_path}")
    print(f"[Saved metrics] {tagged_metrics_path}")


if __name__ == '__main__':
    results = []
    for label, key in EVAL_LOCATIONS:
        results.append(evaluate_location(label, key))
    save_summary_metrics(results)
    save_summary_plot(results)
