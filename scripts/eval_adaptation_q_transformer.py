# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 07:13:18 2026

@author: khali
"""
#%% library part

# Standard library
import os
import pickle
from collections import deque
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local project modules
from eplus_gym.envs.env import AmphitheaterEnv
from eplus_gym.envs.energyplus import (
    _resolve_output_dir
)
from eplus_gym.agents.q_transformer.dqn_agent import Q_transformer


# -----------------------------------------------------------------------------
# Helper utilities                                                            |
# -----------------------------------------------------------------------------

def _ensure_dir(path: str):
    """Create parent directory if it does not yet exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

# -------- training -----------------------------------------------------------

def save_training_metrics(filepath: str, scores, eps_history, steps_array):
    """Persist training arrays to *filepath* (Pickle)."""
    _ensure_dir(filepath)
    with open(filepath, "wb") as f:
        pickle.dump({
            "scores": scores,
            "eps_history": eps_history,
            "steps_array": steps_array,
        }, f)

# 1. Get the absolute path to the main directory 
#    This will be /.../energyplus-gym/    
output_dir = Path(_resolve_output_dir(None))
# 2. recreate gitkeep
output_root = Path(_resolve_output_dir(None))  # your resolver
(output_root / ".gitkeep").touch(exist_ok=True)

checkpoint_dir = _resolve_output_dir(None, default_name="runs/checkpoints")
# --- NEW: where to save figures ---
plots_dir = Path(r"C:\Projects\energyplus-gym-work\energyplus-gym\runs\plots")
plots_dir.mkdir(parents=True, exist_ok=True)

def _save_close(fig, fname: str):
    out = plots_dir / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved plot] {out}")


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


#%%Deployment Q-transformer (Adaptation test)
n_episodes=50

env_config = {
    "output": str(output_dir),
    "location": "luxembourg",

    # 跟 train 一致，這樣 eval 才是評估 penalty 版模型
    "enable_control_penalty": True,
    "w_switch": 0.3,
    "w_du_sat": 0.04,
    "w_du_oac": 0.04,
    "enable_iaq_aware_oac": False,
    "iaq_co2_threshold": 900.0,
    "iaq_oac_penalty_scale": 0.25,
    "iaq_oac_energy_scale": 1.0,

    "enable_pre_occupancy_comfort": False,
    "w_precomfort": 0.3,
    "enable_pre_occupancy_relaxation": False,
    "pre_occ_ahu_energy_scale": 0.7,
    "pre_occ_sat_energy_scale": 0.7,
    "pre_occ_oac_energy_scale": 0.9,
    "pre_occ_sat_penalty_scale": 0.75,
}

env = AmphitheaterEnv(
    env_config=env_config,
    new_begin_month=1, new_end_month=1, new_begin_day=1, new_end_day=10,
    train=False)

env.runner_config.csv = True

MODEL_LABEL = "Q-Transformer"
# Create and load your DQN agent
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
    eps_dec=0.9 * n_episodes * 366,
    chkpt_dir=checkpoint_dir,
    algo='Q-transformer',
    env_name='energyplus'
)

agent.load_models()
agent.q_eval.eval()

scores = []
action_temp = []
action_flow = []
action_proxy_cost = []
ahu_cmd_series = []

# --------------- Run the environment once, collect rewards ---------------
state_window = deque(maxlen=96)
done = False
observation = env.reset()[0]
state_window.append(observation)
score = 0

while not done:
    if len(state_window) < 96:
        action_idx = np.random.choice(agent.action_space)
        count_score = False
    else:
        obs_seq = np.array(state_window, dtype=np.float32)
        action_idx = agent.choose_action_test(obs_seq)
        count_score = True
    
    observation_, reward, terminated, truncated, _ = env.step(action_idx)
    action = env.valid_actions[action_idx]  # discrete -> continuous

    if count_score:
        score += reward
    done = terminated or truncated
    observation = observation_
    state_window.append(observation)
    
    # record command AHU state (0/1)
    ahu_cmd_series.append(int(action[2]))
    action_proxy_cost.append(
        (action[0] / 7) * action[2]
        + (action[1] / 7) * action[2]
        + action[2] / 5
    )
    
    # Convert to real-world scale (optional if you need it)
    action_temp.append(
        env._rescale(action[0] * action[2], range1=(0, 8-1), range2=[15, 30])
    )
    action_flow.append(
        env._rescale(action[1] * action[2], range1=(0, 8-1), range2=[0.3, 5])
    )
    scores.append(reward)

print("Score (entire run): ", score)
env.close()
    


# -------------------------------------------------------------
#  CONFIGURATION  – edit here if you want different colours etc.
# -------------------------------------------------------------
CSV_PATH = Path(output_dir) / "eplusout.csv"

DATE_SLICE = slice(96, 96*17)             # one week -> 672 rows
#DATE_SLICE = slice(192, 288)             # one day
SHORT_CYCLE_MIN_STEPS = 4
ahu_cmd_array = np.array(ahu_cmd_series, dtype=int)
ahu_cmd_window = ahu_cmd_array[DATE_SLICE]
n_switch_cmd = np.sum(ahu_cmd_window[1:] != ahu_cmd_window[:-1])
print("AHU switching count (command):", n_switch_cmd)

SMOOTH_WIN = 15                          # Savitzky–Golay window (odd)
SMOOTH_POLY= 3                           # Savitzky–Golay poly order
FIGSIZE    = (50, 7)

# matplotlib global style
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold",
                     # NEW — bigger tick labels, axis labels and legend text
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.labelsize":  20,
    "legend.fontsize": 20,})
plt.rcParams["xtick.labelsize"] = 20   # bigger numbers on x-axes
plt.rcParams["ytick.labelsize"] = 20   # bigger numbers on y-axes
# -------------------------------------------------------------
#  LOAD & PREP DATA
# -------------------------------------------------------------

data = pd.read_csv(CSV_PATH)
# Fix datetime column
data["Date/Time"] = (
    "2020/" + data["Date/Time"].str.strip()          # add dummy year
                .str.replace(r"\s+", " ", regex=True)
                .str.replace("24:00:00", "23:59:59")
)
data["Date/Time"] = pd.to_datetime(
    data["Date/Time"], format="%Y/%m/%d %H:%M:%S"
).rename("datetime")
data.set_index("Date/Time", inplace=True)

# Slice one week
week = data.iloc[DATE_SLICE]

# Extract columns ------------------------------------------------
zoneT = week["TZ_AMPHITHEATER:Zone Mean Air Temperature [C](TimeStep)"]
htgThr= week["HTG HVAC 1 ADJUSTED BY 1.1 F:Schedule Value [](TimeStep)"]
clgThr= week["CLG HVAC 1 ADJUSTED BY 0 F:Schedule Value [](TimeStep)"]
oat   = week["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"]
occ   = week["MAISON DU SAVOIR AUDITORIUM OCC:Schedule Value [](TimeStep)"]
ahu   = week["AHUS ONOFF:Schedule Value [](TimeStep)"]
elec  = week["Electricity:HVAC [J](TimeStep)"]
heat  = week["Heating:DistrictHeatingWater [J](TimeStep)"]
temp_sp = action_temp[DATE_SLICE]
flow_sp = action_flow[DATE_SLICE]
action_proxy_energy_cost = float(np.sum(action_proxy_cost[DATE_SLICE]))
total_electricity_hvac = float(elec.sum())
total_district_heating = float(heat.sum())

# actual AHU switching from EnergyPlus output
ahu_actual_array = ahu.to_numpy(dtype=int)
n_switch_actual = np.sum(ahu_actual_array[1:] != ahu_actual_array[:-1])

print("AHU switching count (actual state):", n_switch_actual)

co2 = week["TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)"]
occupied_mask = occ.to_numpy() > 0
zoneT_array = zoneT.to_numpy()
htg_array = htgThr.to_numpy()
clg_array = clgThr.to_numpy()
co2_array = co2.to_numpy()

occupied_temp_violation = int(np.sum(
    occupied_mask & ((zoneT_array < htg_array) | (zoneT_array > clg_array))
))
occupied_co2_violation = int(np.sum(
    occupied_mask & (co2_array > 1000.0)
))
short_cycle_count = _count_short_cycles(
    ahu_actual_array,
    min_stable_steps=SHORT_CYCLE_MIN_STEPS,
)

occupied_transition_mask = occupied_mask[1:] & occupied_mask[:-1]
(
    sat_total_variation_occ,
    sat_avg_abs_change_occ,
    sat_max_jump_occ,
) = _control_variation_metrics(temp_sp, occupied_transition_mask)
(
    oa_total_variation_occ,
    oa_avg_abs_change_occ,
    oa_max_jump_occ,
) = _control_variation_metrics(flow_sp, occupied_transition_mask)

print("occupied_temp_violation:", occupied_temp_violation)
print("occupied_co2_violation:", occupied_co2_violation)
print("action_proxy_energy_cost:", action_proxy_energy_cost)
print("total_electricity_hvac [J]:", total_electricity_hvac)
print("total_district_heating [J]:", total_district_heating)
print(
    f"short_cycle_count (<{SHORT_CYCLE_MIN_STEPS} timesteps, actual AHU):",
    short_cycle_count,
)
print("SAT total variation (occupied only):", sat_total_variation_occ)
print("SAT average absolute change (occupied only):", sat_avg_abs_change_occ)
print("SAT maximum jump (occupied only):", sat_max_jump_occ)
print("OA flow total variation (occupied only):", oa_total_variation_occ)
print("OA flow average absolute change (occupied only):", oa_avg_abs_change_occ)
print("OA flow maximum jump (occupied only):", oa_max_jump_occ)

# -------------------------------------------------------------
#  1.  TEMPERATURE & SET-POINTS
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(zoneT, label="Zone T", lw=2)
ax.plot(oat,   label="Outdoor T", lw=2, color="tab:green")
ax.plot(htgThr, label="Heating threshold", ls="--", color="tab:red")
ax.plot(clgThr, label="Cooling threshold", ls=":",  color="tab:red")
ax.step(zoneT.index, temp_sp, label="Supply-air T set-point", color="tab:orange")
# --- bold “DDQN Bi-LSTM” label – choose the spot yourself ----
ax.text(
    0.3, 0.3,                # (x, y) in axes fraction: 0–1 left→right / bottom→top
    MODEL_LABEL,
    transform=ax.transAxes,    # interpret coords in axes space
    fontsize=25,
    fontweight="bold",
)
ax.set(
    ylabel="Temperature (°C)",
)

ax.xaxis.set_major_formatter(
    plt.matplotlib.dates.DateFormatter("%b %d  %H:%M")
)
plt.xticks(rotation=0)

# Twin axes for occupancy & AHU
ax_occ = ax.twinx()
ax_occ.step(zoneT.index, occ*399, label="Occupancy", lw=1.8, color="k")
ax_occ.set_ylabel("Occupants")
ax_occ.set_ylim(0, 400)

ax_ahu = ax.twinx()
ax_ahu.spines.right.set_position(("axes", 1.08))
ax_ahu.step(zoneT.index, ahu, label="AHU state", ls="--", color="grey")
ax_ahu.set_yticks([0, 1]); ax_ahu.set_ylabel("AHU")

ax_ahu.spines["right"].set_visible(True)
ax_ahu.spines["right"].set_color("black")     # darker line
ax_ahu.spines["right"].set_linewidth(1.2)     # a bit thicker

ax_flow = ax.twinx()
ax_flow.spines.right.set_position(("axes", 1.04))
ax_flow.step(zoneT.index, flow_sp, label="OAC mass flow", lw=1.6, color="tab:cyan")
ax_flow.set_ylabel("Outdoor-air flow rate (kg/s)")

ax_flow.spines["right"].set_visible(True)
ax_flow.spines["right"].set_color("black")
ax_flow.spines["right"].set_linewidth(1.2)
# Collect legends from all axes
lines, labels = [], []
for a in (ax, ax_occ, ax_ahu, ax_flow):
    L, lab = a.get_legend_handles_labels()
    lines += L; labels += lab
ax.legend(lines, labels, loc="upper left")

# 1) remove the white grid lines completely
for a in (ax_occ, ax_ahu, ax_flow):
    a.grid(False)

# 2) shrink right margin so no wasted white area
fig.subplots_adjust(right=0.82)


LEGEND_X = 0.35   # 0 = left edge of axes, 1 = right edge, >1 = outside right
LEGEND_Y = 0.95   # 0 = bottom edge,   1 = top edge,   >1 = above axes
# merge legends
lines, labels = [], []
for a in (ax, ax_occ, ax_ahu, ax_flow):
    L, lab = a.get_legend_handles_labels()
    lines += L; labels += lab

# 8 items → 4 columns ⇒ 2 rows
ax.legend(
    lines, labels,
    loc="upper center",            # anchor at centre-top …
    bbox_to_anchor=(LEGEND_X, LEGEND_Y),    # … just above the axes
    ncol=2,                        # 4 columns → 2 rows
    fontsize=22,
    frameon=False                  # optional: no box edge
)
# ... legend code ...

# -------------------------------------------------------------
#  Add rose-tinted background for the deployment period
# -------------------------------------------------------------
split_dt = pd.Timestamp("2020-01-08 00:00:00")   # start of 8 Jan

ax.axvspan(split_dt, zoneT.index[-1],           # ← from split to end
           facecolor="#e6f7e6", alpha=1,     # ← light rose
           zorder=-5)                           # ← behind every artist



#--------------------------------------------------------------------------


ax.set_xlim(zoneT.index[0], zoneT.index[-1])
ax.margins(x=0)
plt.text(0.86, 0.99, f'Total Score: {score:.2f}', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')

# plt.show()
_save_close(fig, "eval_adaptation_q_transformer_temperature_setpoints.png")



# -------------------------------------------------------------
#  2.  ENERGY CONSUMPTION
# -------------------------------------------------------------
elec = week["Electricity:HVAC [J](TimeStep)"]
heat = week["Heating:DistrictHeatingWater [J](TimeStep)"]

fig, ax1 = plt.subplots(figsize=FIGSIZE)
ax1.step(elec.index, elec, color="tab:blue", label="Electricity: HVAC [J]")
ax1.set_ylabel("Electricity [J]")
ax1.xaxis.set_major_formatter(
    plt.matplotlib.dates.DateFormatter("%b %d  %H:%M")
)
plt.xticks(rotation=0)

ax2 = ax1.twinx()
ax2.step(heat.index, heat, color="tab:orange", label="District Heating [J]")
ax2.set_ylabel("District Heating [J]")


lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines+lines2, labels+labels2, loc="upper left",fontsize=14)
# 1) remove the white grid lines completely
for a in (ax1,ax2):
    a.grid(False)
# Annotate totals
tot_elec = elec.sum(); tot_heat = heat.sum()
ax1.text(0.86, 0.95, f"∑ Elec: {tot_elec/1e9:.2f} GJ",
         transform=ax1.transAxes,fontsize=20)
ax1.text(0.86, 0.88, f"∑ Heat: {tot_heat/1e9:.2f} GJ",
         transform=ax1.transAxes,fontsize=20)
ax1.set_xlim(elec.index[0], elec.index[-1])
ax1.margins(x=0)
ax1.text(
    0.3, 0.3,            # ← move left↔right / down↕up as you wish
    MODEL_LABEL,
    transform=ax1.transAxes,
    fontsize=25,
    fontweight="bold",
)
# merge legends
lines, labels   = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2, labels + labels2,
           loc="upper left",            # anchor-point = upper-left corner
           bbox_to_anchor=(0.3, LEGEND_Y),# move 8 % of axes-width to the LEFT
           fontsize=22, frameon=False)  # no box edge if you prefer
# -------------------------------------------------------------
#  Add rose-tinted background for the deployment period
# -------------------------------------------------------------
split_dt = pd.Timestamp("2020-01-08 00:00:00")   # start of 8 Jan

ax1.axvspan(split_dt, zoneT.index[-1],           # ← from split to end
           facecolor="#e6f7e6", alpha=1,     # ← light rose
           zorder=-5)                           # ← behind every artist
_save_close(fig, "eval_adaptation_q_transformer_energy.png")
# -------------------------------------------------------------
#  3.  CO₂ CONCENTRATION
# -------------------------------------------------------------
co2 = week["TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)"]

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(co2.index, co2, color="tab:purple")
ax.set(
    ylabel="CO2 [ppm]",
)
ax.xaxis.set_major_formatter(
    plt.matplotlib.dates.DateFormatter("%b %d  %H:%M")
)
ax.text(
    0.3, 0.3,            # same idea—adjust to suit this plot
    MODEL_LABEL,
    transform=ax.transAxes,
    fontsize=25,
    fontweight="bold",
)
plt.xticks(rotation=0)

tot_co2 = co2.sum()
# -------------------------------------------------------------
#  Add rose-tinted background for the deployment period
# -------------------------------------------------------------
split_dt = pd.Timestamp("2020-01-08 00:00:00")   # start of 8 Jan

ax.axvspan(split_dt, zoneT.index[-1],           # ← from split to end
           facecolor="#e6f7e6", alpha=1,     # ← light rose
           zorder=-5)                           # ← behind every artist
ax.set_xlim(co2.index[0], co2.index[-1])
ax.margins(x=0)
# plt.show()
_save_close(fig, "eval_adaptation_q_transformer_co2.png")    
   
