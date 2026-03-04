import dill
import numpy as np
import matplotlib.pyplot as plt
import data_visualization as dv
import src.model_params as param
import src.constants as const
import src.plot_functions as pf
from copy import deepcopy
from pathlib import Path
import polars as pl
from src.control_model import (
    GTOControlPolicy,
    GTODynamics,
    StateEstimator,
    DualModelSimulator,
)
import importlib

importlib.reload(param)

pl.Config.set_tbl_rows(50)

wheel = dv.ColorWheel()
dv.set_plot_style("cashaback_dark.mplstyle")
"""

Comparing models using the differential game theory model.

The GT model is just an LQG if partner_knowledge = False.
It's also a bimanual model if parnter_knowledge = True and alpha = 1.0.
It's completely selfish if partner_knowledge = True and alpha = 0.0.

"""


# %% Functions
def create_single_trial_df(
    trial, model, state_mapping, condition, experiment_trial_number
):
    df = pl.DataFrame()
    df = df.with_columns(
        model_name=np.array([model.model_name] * len(model.timesteps)).squeeze(),
        condition=np.array([condition] * len(model.timesteps)).squeeze(),
        trial=np.array([trial] * len(model.timesteps)).squeeze(),
        experiment_trial_number=np.array(
            [experiment_trial_number] * len(model.timesteps)
        ).squeeze(),
        timepoint=model.timesteps * 10,  # x10 for ms
        jump_type=np.array(
            [model.perturbation_states[0]] * len(model.timesteps)
        ).squeeze(),
        jump_time=np.array([model.jump_time] * len(model.timesteps)).squeeze(),
        jump_back_time=np.array(
            [model.jump_back_time] * len(model.timesteps)
        ).squeeze(),
        alpha=np.array(
            [model.p1_policy.alpha] * len(model.timesteps), dtype=float
        ).squeeze(),
        partner_knowledge=np.array(
            [model.partner_knowledge] * len(model.timesteps)
        ).squeeze(),
        p1_applied_force=np.append(np.array(model.p1_applied_force), np.nan).squeeze(),
        p2_applied_force=np.append(np.array(model.p2_applied_force), np.nan).squeeze(),
        u1x=np.append(model.u1[:, 0], np.nan).squeeze(),
        u1y=np.append(model.u1[:, 1], np.nan).squeeze(),
        u2x=np.append(model.u2[:, 0], np.nan).squeeze(),
        u2y=np.append(model.u2[:, 1], np.nan).squeeze(),
    )
    # TODO add time_from_probe_onset column
    df = df.with_columns(time_from_jump=pl.col("timepoint").sub(model.jump_time))
    for key, val in state_mapping.items():
        df = df.with_columns(pl.Series(key, model.x[:, val].squeeze()))
    return df


def set_matrices(condition):
    if "solo" in condition:
        A = param.A_solo
        A_probe = param.A_probe_solo
        B2 = np.zeros_like(param.B2)  # turn off partner
        Q2 = np.zeros_like(param.Q2)
    else:
        A = param.A_joint
        A_probe = param.A_probe_joint
        B2 = param.B2
        Q2 = param.Q2
    return A, A_probe, B2, Q2


# @profile()
def set_model_objects(A, A_probe, B2, Q2, alpha, partner_knowledge):
    dynamics = GTODynamics(
        A=deepcopy(A),
        A_probe=deepcopy(A_probe),
        B1=deepcopy(param.B1),
        B2=deepcopy(B2),
        C1=deepcopy(param.C1),
        C2=deepcopy(param.C2),
        x0=deepcopy(param.x0),
        N=deepcopy(param.N),
        state_mapping=deepcopy(param.state_mapping),
        h=deepcopy(param.h),
        sensory_delay=param.sensory_delay_steps,
    )

    p1_state_estimator = StateEstimator(
        dynamics=dynamics,
        player=1,
        W=deepcopy(param.W1_cov),
        V=deepcopy(param.V1_cov),
        sensor_noise_arr=param.MEASUREMENT_NOISE_MOD1,
        internal_model_noise_arr=param.MEASUREMENT_NOISE_MOD1,
    )
    p2_state_estimator = StateEstimator(
        dynamics=dynamics,
        player=2,
        W=deepcopy(param.W2_cov),
        V=deepcopy(param.V2_cov),
        sensor_noise_arr=param.MEASUREMENT_NOISE_MOD2,
        internal_model_noise_arr=param.MEASUREMENT_NOISE_MOD2,
    )

    p1_policy = GTOControlPolicy(
        player=1,
        dynamics=dynamics,
        Q_self=deepcopy(param.Q1),
        Q_partner=deepcopy(Q2),
        R_self=deepcopy(param.R11) * (1 - alpha),
        R_self_partner=deepcopy(param.R12) * alpha,
        R_partner_self=deepcopy(param.R21) * alpha,
        R_partner=deepcopy(param.R22) * (1 - alpha),
        alpha=alpha,
        partner_knowledge=partner_knowledge,
    )
    p2_policy = GTOControlPolicy(
        player=2,
        dynamics=dynamics,
        Q_self=deepcopy(Q2),
        Q_partner=deepcopy(param.Q1),
        R_self=deepcopy(param.R22) * (1 - alpha),
        R_self_partner=deepcopy(param.R21) * alpha,
        R_partner_self=deepcopy(param.R12) * alpha,
        R_partner=deepcopy(param.R11) * (1 - alpha),
        alpha=alpha,
        partner_knowledge=partner_knowledge,
    )

    return dynamics, p1_state_estimator, p2_state_estimator, p1_policy, p2_policy


# %%
SAVE = True
SAVE_PATH = Path(r"data/models")
# %% Set up constants and matrices
# Timesteps
model_names = [
    "exp1_ofc",
    "exp1_gto",
    "exp1_gto_joint_cost",
    "exp1_gto_wtd_joint_cost",
]

partner_knowledges = [
    False,
    True,
    True,
    True,
]
alphas = [
    0,
    0,
    0.5,
    0.3
]
experimental_condition = [
    [[0.02, 0.02], ["rtx", "ltx"]],  # joint same jump
    [[0.02, 0.00], ["rtx", "ltx"]],  # joint self jump
    [[0.00, 0.02], ["rtx", "ltx"]],  # joint partner jump
    [[0.02, 0.02], ["rtx", "ltx"]],  # solo same jump
    [[0.02, 0.00], ["rtx", "ltx"]],  # solo self jump
    [[0.00, 0.02], ["rtx", "ltx"]],  # solo partner jump
]
NUM_TRIALS = 100
assert len(experimental_condition) == len(const.condition_names)
regular_df_list = []
probe_df_list = []
all_models = {k: [].copy() for k in model_names}
for i, partner_knowledge in enumerate(partner_knowledges):
    exp_trial_num = 0
    for j, (perturbation_size, perturbation_target) in enumerate(
        experimental_condition
    ):
        exp_trial_num += 1
        print(perturbation_size, perturbation_target)

        A, A_probe, B2, Q2 = set_matrices(const.condition_names[j])

        # NOTE ONLY CARE ABOUT PARTNER TARGET DURING JOINT CONDITIONS
        alpha = alphas[i] if const.condition_names[j].startswith("joint") else 0.0

        (dynamics, p1_state_estimator, p2_state_estimator, p1_policy, p2_policy) = (
            set_model_objects(A, A_probe, B2, Q2, alpha, partner_knowledge)
        )

        dynamics.discretize()
        dynamics.augment()

        p1_state_estimator.set_linear_kalman_gain()
        p2_state_estimator.set_linear_kalman_gain()

        p1_policy.discretize()
        p1_policy.care_about_partner_Q()
        p2_policy.discretize()
        p2_policy.care_about_partner_Q()
        
        # Make sure they know the others after caring about each other 
        p1_policy.Q_partner = p2_policy.Q_self
        p2_policy.Q_partner = p1_policy.Q_self

        p1_policy.solve_ricatti_discrete()
        p2_policy.solve_ricatti_discrete()

        model = DualModelSimulator(
            model_name=model_names[i],
            movement_type="reaching",
            dynamics=dynamics,
            p1_policy=p1_policy,
            p2_policy=p2_policy,
            p1_state_estimator=p1_state_estimator,
            p2_state_estimator=p2_state_estimator,
            process_noise_value=0.01,
            perturbation_onset_distance=0.015,
            perturbation_states=perturbation_target,
            perturbation_distances=perturbation_size,
            probe_duration=param.probe_duration_steps,
        )
        all_models[model_names[i]].append(deepcopy(model))
        if SAVE:
            with open(SAVE_PATH / f"model_object_{model_names[i]}_{const.condition_names[j]}.pkl", 'wb') as f:
                dill.dump(model,f)

        for k in range(NUM_TRIALS):
            model.run_simulation(probe_trial=False)
            regular_df = create_single_trial_df(
                trial=k,
                model=model,
                state_mapping=param.state_mapping,
                condition=const.condition_names[j],
                experiment_trial_number=exp_trial_num,
            )
            regular_df_list.append(regular_df)

            model.run_simulation(probe_trial=True)
            probe_df = create_single_trial_df(
                trial=k,
                model=model,
                state_mapping=param.state_mapping,
                condition=const.condition_names[j],
                experiment_trial_number=exp_trial_num,
            )
            probe_df_list.append(probe_df)
    regular_df = pl.concat(regular_df_list)
    probe_df = pl.concat(probe_df_list)

if SAVE:
    with open(SAVE_PATH / f"regular_model_jump_df.pkl", "wb") as f:
        dill.dump(regular_df.with_row_index(), f)

    with open(SAVE_PATH / f"probe_model_jump_df.pkl", "wb") as f:
        dill.dump(probe_df.with_row_index(), f)

# %% Plot Feedback Gains
for i, model_name in enumerate(model_names[:2]):
    fig = dv.AutoFigure("abc;def", figsize=(9, 9), dpi=120)
    axes = list(fig.axes.values())
    for j, condition in enumerate(const.condition_names):
        model = all_models[model_name][j]
        model.run_simulation(probe_trial=True)  # Running so I have jump_step
        pf.plot_feedback_gains(
            ax=axes[j],
            F1=model.p1_policy.F,
            F2=model.p2_policy.F,
            state_mapping=param.state_mapping,
            state_labels=["ccx", "rtx", "ltx"],
            timesteps=model.timesteps[:-1],
        )
        axes[j].set_title(condition)
        # axes[j].set_ylim(-10,18000)
        axes[j].axvline(model.jump_step)
        axes[j].axvline(model.jump_back_step)
        axes[j].axvline(model.jump_step + 0.12, c=wheel.grey)
        axes[j].axvline(model.jump_back_step + 0.12, c=wheel.grey)

# %% Plot xy states
for i, model_name in enumerate(model_names):
    fig = dv.AutoFigure("abc;def", figsize=(9, 9), dpi=120)
    axes = list(fig.axes.values())
    for j, condition in enumerate(const.condition_names):
        model = all_models[model_name][j]
        model.run_simulation(probe_trial=False)
        axes[j].plot(
            model.x[:, param.state_mapping["rhx"]],
            model.x[:, param.state_mapping["rhy"]],
            c=const.self_color,
        )
        axes[j].plot(
            model.x1_post[:, param.state_mapping["ccx"]],
            model.x1_post[:, param.state_mapping["ccy"]],
            c=const.cc_color,
        )
        axes[j].plot(
            model.x[:, param.state_mapping["lhx"]],
            model.x[:, param.state_mapping["lhy"]],
            c=const.partner_color,
        )
        axes[j].scatter(
            model.x[0, param.state_mapping["rtx"]],
            model.x[0, param.state_mapping["rty"]],
            facecolor="None",
            edgecolors="grey",
        )
        axes[j].set_title(condition, fontsize=9)
    # axes[j].set_xlim(-.2,.2)
    # axes[j].set_ylim(-.01,.3)

# fig.fig.suptitle(model_name)

# %% Plot Y-pos over time
for i, model_name in enumerate(model_names):
    fig = dv.AutoFigure("abc;def", figsize=(9, 9), dpi=120)
    axes = list(fig.axes.values())
    for j, condition in enumerate(const.condition_names):
        model = all_models[model_name][j]
        pf.plot_states_over_time(
            ax=axes[j],
            x=model.x,
            state_mapping=param.state_mapping,
            state_labels=["rhy", "lhy", "ccy"],
            linestyles=["-", "--", ":"],
        )
        axes[j].set_title(condition)
        axes[j].set_ylabel("Y-Position (m)")
# %% Plot X-pos over time
for i, model_name in enumerate(model_names):
    fig = dv.AutoFigure("abc;def", figsize=(9, 9), dpi=120)
    axes = list(fig.axes.values())
    for j, condition in enumerate(const.condition_names):
        model = all_models[model_name][j]
        pf.plot_states_over_time(
            ax=axes[j],
            x=model.x,
            state_mapping=param.state_mapping,
            state_labels=["rhx", "lhx", "ccx"],
            linestyles=["-", "-", ":"],
        )
        axes[j].set_title(condition)
        axes[j].set_ylabel("X-Position (m)")
# %% Plot X-Velocity over time
for i, model_name in enumerate(model_names):
    fig = dv.AutoFigure("abc;def", figsize=(9, 9), dpi=120)
    axes = list(fig.axes.values())
    for j, condition in enumerate(const.condition_names):
        model = all_models[model_name][j]
        pf.plot_states_over_time(
            ax=axes[j],
            x=model.x,
            state_mapping=param.state_mapping,
            state_labels=["rhvx", "lhvx"],
            linestyles=["-", "-", ":"],
        )
        axes[j].set_title(condition)
        axes[j].set_ylabel("X-Velocity (m/s)")
        # axes[j].set_ylim(-0.01,0.2)
# %% Plot force responses
ylim = (-8, 8)
for i, model_name in enumerate(model_names):
    fig = dv.AutoFigure("abc;def", figsize=(9, 9), dpi=120)
    axes = list(fig.axes.values())
    for j, condition in enumerate(const.condition_names):
        model = all_models[model_name][j]
        axes[j].plot(
            model.timesteps[:-1],
            model.p1_applied_force,
            color=wheel.rak_red,
            label="RH Lateral Force",
        )
        axes[j].plot(
            model.timesteps[:-1],
            model.p2_applied_force,
            color=wheel.rak_blue,
            label="LH Lateral Force",
            ls="--",
        )
        axes[j].axvline(x=model.jump_time, ls="--", color=wheel.grey)
        axes[j].axvline(x=model.jump_back_time, ls="--", color=wheel.grey)

        # PLot involuntary region
        axes[j].fill_betweenx(
            np.arange(ylim[0], ylim[1], 0.01),
            model.jump_time + 0.18,  # add 180ms, relative to total time
            model.jump_time + 0.23,  # add 230ms, relative to total time
            facecolor=wheel.lighten_color(wheel.light_grey, 0.75),
            alpha=0.1,
        )
        axes[j].text(
            model.jump_time + 0.205,
            ylim[1],
            "Involuntary",
            rotation=90,
            va="top",
            ha="center",
            color=wheel.grey,
            fontsize=10,
        )

        # axes[j].set_xticks(np.arange(0,+0.1,0.1))
        axes[j].set_ylabel("Applied Force Into Channel")
        axes[j].set_xlabel("Time (ms)")
        axes[j].set_title(condition)
        # axes[j].set_xlim(model.jump_time-0.02,0.7)
        axes[j].set_ylim(ylim[0], ylim[1])

        axes[j].legend()
# %% One plot all condition force traces
ylim = (-5, 5)
for i, model_name in enumerate(model_names):
    fig = dv.AutoFigure("a", figsize=(9, 9), dpi=100)
    ax = fig.axes["a"]
    for j, condition in enumerate(CONDITIONS):
        df = probe_df.filter(
            pl.col("model_name") == model_name,
            pl.col("condition") == condition,
            pl.col("trial") == 1,
        )
        jump_time = df["jump_time"][0]
        dff = df.filter(pl.col("time_from_jump").is_between(-50, 400))
        ax.plot(
            dff["time_from_jump"],
            dff["p1_applied_force"],
            color=const.condition_colors_dark[j],
            label=condition,
        )
        print(df["jump_time"][0])
        # peakx = model.timesteps[60]
        # peaky = np.max(2*model.applied_force1)
        # ax.text(peakx, peaky,condition,
        #         transform=ax.transData, color=const.condition_colors_dark[j],
        #         va='bottom', ha='center', fontsize=6.5, fontweight='bold')
        # PLot involuntary region
        ax.axvline(180)
        ax.axvline(230)

        ax.set_ylabel("Applied Force Into Channel")
        ax.set_xlabel("Time (ms)")
        # ax.set_title(condition)
        # ax.set_xlim(jump_step-0.02, jump_step+0.4)
        # ax.set_ylim(ylim[0],ylim[1])
        handles, labels = fig.axes["a"].get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            ncols=1,
            loc=(0.05, 0.7),
            labelcolor="linecolor",
            frameon=False,
        )
    ax.set_title(model_name)