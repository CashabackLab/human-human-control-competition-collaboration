import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.constants as const
import data_visualization as dv
import src.helper_functions as hf
import polars as pl
from src.dyad import Dyad
import src.constants as const
import warnings
import matplotlib.animation as animation
import matplotlib.patches as patches

wheel = dv.ColorWheel()


def get_target_patches(
    ax,
    condition,
    self_color=const.self_color,
    partner_color=const.partner_color,
    flip_p2=True,
    border_width=1.0,
    edge_alpha=1.0,
    face_alpha=1.0,
    jump_direction=1.0,
    target_width=0.0175,
    target_height=0.01,
    xcenter=0.54,
    ycenter=0.28,
    jump_distance=0.02,
    transform=None,
):
    condition = condition.lower()
    if "self" in condition or "p1" in condition:
        self_jump = jump_distance / 2
        partner_jump = -jump_distance / 2
    elif "partner" in condition or "p2" in condition or "other" in condition or "opponent" in condition:
        self_jump = -jump_distance / 2
        partner_jump = jump_distance / 2
    elif "same" in condition or "both" in condition or "collaborative" in condition:
        self_jump = 0
        partner_jump = 0
    else:
        raise ValueError(
            "'condition' string should contain 'self_jump', 'p1_jump', 'partner_jump', 'p2_jump', or 'same_jump'"
        )

    if transform is None:
        transform = ax.transData

    self_target = patches.Rectangle(
        (
            xcenter - target_width / 2 + self_jump * jump_direction,
            ycenter - target_height / 2,
        ),
        target_width,
        target_height,
        linewidth=border_width,
        color=self_color,
        transform=transform,
    )

    partner_target = patches.Rectangle(
        (
            xcenter - target_width / 2 + partner_jump * jump_direction,
            ycenter - target_height / 2,
        ),
        target_width,
        target_height,
        linewidth=border_width,
        facecolor="none",
        edgecolor=partner_color,
        transform=transform,
    )
    return self_target, partner_target


def plot_jump_arrows(
    ax,
    x,
    y,
    dx,
    dy,
    arrow_yshift,
    condition,
    self_arrow_style=None,
    partner_arrow_style=None,
    transform=None,
    clip_on=False,
    arrow_head_width=0.1,
    arrow_head_length=0.012,
    arrow_lw=1.5,
    tail_width=1.0,
):
    if transform is None:
        transform = ax.transData

    if self_arrow_style is None:
        self_arrow_style = dict(
            head_width=arrow_head_width,
            head_length=arrow_head_length,
            color=const.self_color,
            lw=arrow_lw,
            length_includes_head=True,
            shape="right",
            width=tail_width,
            alpha=1,
        )
    if partner_arrow_style is None:
        partner_arrow_style = dict(
            head_width=arrow_head_width,
            head_length=arrow_head_length,
            facecolor="none",
            edgecolor=const.partner_color,
            lw=arrow_lw,
            width=tail_width,
            length_includes_head=True,
            shape="left",
            alpha=1,
            fill=False,
        )
    condition = condition.lower()
    if "same" in condition or "both" in condition or "collaborative" in condition:
        ax.arrow(x, y, dx, dy, transform=transform, **self_arrow_style, clip_on=clip_on)
        ax.arrow(
            x,
            y - arrow_yshift,
            dx,
            dy,
            transform=transform,
            **partner_arrow_style,
            clip_on=clip_on,
        )
    elif "p1" in condition or "self" in condition:
        ax.arrow(x, y, dx, dy, transform=transform, **self_arrow_style, clip_on=clip_on)
    elif "p2" in condition or "partner" in condition or "other" in condition or "opponent" in condition:
        ax.arrow(
            x,
            y + arrow_yshift,
            dx,
            dy,
            transform=transform,
            **partner_arrow_style,
            clip_on=clip_on,
        )
    else:
        raise ValueError(f"Condition: {condition}, is not valid")


def target_legend(
    ax,
    inset_x,
    inset_y,
    inset_w,
    inset_h,
    condition_name,
    label_name,
    label_color,
    xpos,
    ypos,
    target_width,
    target_height,
    jump_distance,
    fontsize=7,
    target_pad=0.01,
    arrow_lw=1.5,
    arrow_head_width=0.1,
    arrow_head_length=0.012,
    arrow_shape="left",
    inset_axis_off=True,
    arrow_yshift=0.1,
    tail_width=0.015,
    transform=None,
):
    ax_inset = ax.inset_axes([inset_x, inset_y, inset_w, inset_h], transform=transform)
    ax_inset.text(
        0.5,
        1,
        label_name,
        color=label_color,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=fontsize,
    )
    self_target, partner_target = get_target_patches(
        ax_inset,
        xcenter=xpos,
        ycenter=ypos,
        jump_distance=jump_distance,
        target_height=target_height,
        target_width=target_width,
        condition=condition_name,
    )
    self_arrow_style = dict(
        head_width=arrow_head_width,
        head_length=arrow_head_length,
        color=const.self_color,
        lw=arrow_lw,
        length_includes_head=True,
        shape="right",
        alpha=1,
        width=tail_width,
    )
    partner_arrow_style = dict(
        head_width=arrow_head_width,
        head_length=arrow_head_length,
        facecolor="none",
        edgecolor=const.partner_color,
        lw=arrow_lw,
        length_includes_head=True,
        shape="left",
        alpha=1,
        width=tail_width,
    )

    ax_inset.add_patch(self_target)
    ax_inset.add_patch(partner_target)
    xjumpshift = jump_distance / 2 if "same" not in condition_name else 0
    plot_jump_arrows(
        ax_inset,
        x=xpos - target_width / 2 + xjumpshift,
        y=ypos - target_height - 0.05,
        dx=target_width,
        dy=0,
        transform=ax_inset.transData,
        arrow_yshift=arrow_yshift,
        condition=condition_name,
        self_arrow_style=self_arrow_style,
        partner_arrow_style=partner_arrow_style,
        clip_on=False,
    )

    ax_inset.set_xlim(0, 1)
    ax_inset.set_ylim(0, 1)
    if inset_axis_off:
        ax_inset.set_axis_off()


def trace_plot(
    dyad: Dyad,
    metric: str,
    perturbations: list[str],
    average_timepoints,
    df_type,
    average_subjects=False,
    figsize=(6, 5),
    dpi=150,
    dyad_num: int = None,
    xlabel="",
    ylabel="",
    xlim=None,
    ylim=None,
    window_line_color="white",
):
    """
    metric: position or force
    """
    # * WARNING
    if any("probe" in x for x in perturbations) and df_type != "probe":
        warnings.warn("Plotting probe trials but not using df_type='probe'")

    col_names = [col for col in dyad._get_columns(metric) if ("Filtered" in col or "center_cursor" in col)]
    mosaic = [
        ["e", "e"],
        ["solo_same_jump", "solo_opposite_jump"],
        ["joint_same_jump", "joint_opposite_jump"],
    ]
    data = {k: 0 for k in col_names}  # Initialize dictionary for later
    timepoints = {k: 0 for k in col_names}

    figs = []
    for j, perturbation in enumerate(perturbations):
        fig = dv.AutoFigure(mosaic=mosaic, figsize=figsize, dpi=dpi, height_ratios=[0.1, 1, 1])
        for condition in const.condition_names:
            df = dyad.get_traces(
                metric=metric,
                perturbation_type=perturbation,
                condition=condition,
                df_type=df_type,
                average_timepoints=average_timepoints,
            )

            for col_name in col_names:
                data[col_name] = hf.groupby_trial_to_numpy(df, col_name).T
                timepoints[col_name] = hf.groupby_trial_to_numpy(df, "timepoint").T

            ax = fig.axes[condition]
            if metric == "pos":
                # Traces
                ax.plot(
                    data[col_names[0]],
                    data[col_names[1]],
                    c=const.self_color,
                    zorder=1,
                    alpha=1,
                )
                ax.plot(
                    data[col_names[4]],
                    data[col_names[5]],
                    c=const.cc_color,
                    zorder=1,
                    alpha=1,
                )
                ax.plot(
                    data[col_names[6]],
                    data[col_names[7]],
                    c="blue",
                    zorder=1,
                    alpha=1,
                )
                ax.plot(
                    data[col_names[2]],  # NOTE the -x for the left hand
                    data[col_names[3]],
                    c=const.partner_color,
                    zorder=1,
                    alpha=1,
                )
                colors = [
                    const.partner_color,
                    const.cc_color,
                    const.self_color,
                ]
                labels = ["Player 2", "Center Cursor", "Player 1"]
                p1_targets, p2_targets = get_target_patches(ax)
                # TODO Fix the target patches
                # ax.add_patch(p1_targets[condition])
                # ax.add_patch(p2_targets[condition])

            elif metric == "force":
                ax.plot(
                    timepoints[col_names[0]],
                    data[col_names[0]],
                    c=const.self_color,
                )
                ax.plot(
                    timepoints[col_names[1]],
                    data[col_names[1]],
                    c=const.partner_color,
                )
                colors = [const.self_color, const.partner_color]
                labels = ["Player 1", "Player 2"]
                ax.set_xticks(np.arange(np.min(timepoints[col_names[0]]), 450, 75))
                ax.axvline(x=0, ls="--", c=wheel.grey)
                ax.axvline(x=180, ls="--", c=window_line_color)
                ax.axvline(x=230, ls="--", c=window_line_color)
                ax.fill_betweenx(
                    np.arange(ylim[0], ylim[1], 0.01),
                    180,
                    230,
                    color=wheel.grey,
                    alpha=0.3,
                )

        for k, (n, ax) in enumerate(fig.axes.items()):
            if n == "e":
                continue
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(f"{const.condition_names[k - 1]}", fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        dv.legend(
            fig.axmain,
            labels=labels,
            colors=colors,
            ncols=len(labels),
            columnspacing=2,
            fontsize=10,
            loc="center",
            bbox_to_anchor=(0.5, 0.975),
        )

        fig.axmain.set_title(f"Dyad {dyad_num} {perturbation}")
        fig.axes["e"].axis("off")
        figs.append(fig)
    return figs


def plot_ricatti_solution(timesteps, P1_sol, P2_sol, state_labels, state_mapping):
    fig = dv.AutoFigure("a;b", figsize=(5, 6))
    ax1 = fig.axes["a"]
    ax2 = fig.axes["b"]
    xstate_ids = [state_mapping.get(key) for key in state_labels if key.find("x") != -1]
    xstate_labels = [key for key in state_labels if key.find("x") != -1]
    ystate_ids = [state_mapping.get(key) for key in state_labels if key.find("y") != -1]
    ystate_labels = [key for key in state_labels if key.find("y") != -1]

    for i in range(len(xstate_ids)):
        idx = xstate_ids[i]
        idy = ystate_ids[i]
        label = state_labels[i]

        ax1.plot(timesteps, P1_sol[:, idx, idx], label=f"P1 x {xstate_labels[i]}")
        ax1.plot(timesteps, P1_sol[:, idy, idy], label=f"P1 y {ystate_labels[i]}", ls="--")

        ax2.plot(timesteps, P2_sol[:, idx, idx], label=f"P2 x {xstate_labels[i]}")
        ax2.plot(timesteps, P2_sol[:, idy, idy], label=f"P2 y {ystate_labels[i]}", ls="--")

    for ax in [ax1, ax2]:
        ax.set_ylabel("P(t)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left")

    ax1.set_title("Player 1 Ricatti Solution")
    ax2.set_title("Player 2 Ricatti Solution")

    fig.remove_figure_borders()
    return fig, ax1, ax2


def plot_all_feedback_gains(M, state_mapping, figsize=(40, 10)):
    M = M[:, : len(state_mapping), : len(state_mapping)]  # Handle augmentation

    if M.shape[1] > 2 and figsize[1] < 20:
        figsize = (40, 40)

    fig, axes = plt.subplots(M.shape[1], M.shape[2], dpi=250, figsize=figsize)

    if M.shape[1] == 2:
        ylabs = ["x", "y"]
    else:
        ylabs = list(state_mapping.keys())
    titles = list(state_mapping.keys())
    c = -1
    for i in range(M.shape[1]):
        for j in range(M.shape[2]):
            c += 1
            axes[i, j].plot(M[:, i, j])
            if i == 0:
                axes[i, j].set_title(titles[j])

            if j == 0:
                axes[i, j].set_ylabel(ylabs[i], fontsize=20)
    return fig, axes


def plot_feedback_gains(timesteps, F1, F2, state_labels, state_mapping, ax=None):
    if ax is None:
        fig = dv.AutoFigure("a;b", figsize=(5, 6))
        ax1 = fig.axes["a"]
        ax2 = fig.axes["b"]

        for label in state_labels:
            if label.find("x") != -1:
                ax1.plot(timesteps, F1[:, 0, state_mapping[label]], label=f"P1 Fx {label}")
                ax2.plot(
                    timesteps,
                    F2[:, 0, state_mapping[label]],
                    label=f"P2 Fx {label}",
                    ls="-.",
                )
            else:
                ax1.plot(
                    timesteps,
                    F1[:, 1, state_mapping[label]],
                    label=f"P1 Fy {label}",
                    ls="--",
                )
                ax2.plot(
                    timesteps,
                    F2[:, 1, state_mapping[label]],
                    label=f"P2 Fy {label}",
                    ls=":",
                )

        for ax in [ax1, ax2]:
            ax.set_ylabel("F(t)")
            ax.set_xlabel("Time (s)")
            ax.legend(loc="upper left")

        ax1.set_title("Player 1 Feedback Gains")
        ax2.set_title("Player 2 Feedback Gains")

        fig.remove_figure_borders()
        return fig, ax1, ax2

    else:
        for label in state_labels:
            if label.find("x") != -1:
                ax.plot(timesteps, F1[:, 0, state_mapping[label]], label=f"P1 Fx {label}")
                ax.plot(
                    timesteps,
                    F2[:, 0, state_mapping[label]],
                    label=f"P2 Fx {label}",
                    ls="-.",
                )
            else:
                ax.plot(
                    timesteps,
                    F1[:, 1, state_mapping[label]],
                    label=f"P1 Fy {label}",
                    ls="--",
                )
                ax.plot(
                    timesteps,
                    F2[:, 1, state_mapping[label]],
                    label=f"P2 Fy {label}",
                    ls=":",
                )

        ax.set_ylabel("F(t)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left")

        return ax


def plot_control_signal(timesteps, u1, u2):
    fig = dv.AutoFigure("a;b", figsize=(5, 6))
    ax1 = fig.axes["a"]
    ax2 = fig.axes["b"]
    ax1.plot(timesteps, u1[:, 0], label=f"P1 u1x")
    ax1.plot(timesteps, u1[:, 1], label=f"P1 u1y", ls="--")
    ax2.plot(timesteps, u2[:, 0], label=f"P2 u2x")
    ax2.plot(timesteps, u2[:, 1], label=f"P2 u2y", ls="--")

    for ax in [ax1, ax2]:
        ax.set_ylabel("u(t)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left")

    ax1.set_title("Player 1 Control Signal")
    ax2.set_title("Player 2 Control Signal")

    fig.remove_figure_borders()
    return fig, ax1, ax2


def plot_xy_position(x, x_post, state_mapping, dual=True, muscle=False, separate_targets=False, ax=None):
    rhx_id, rhy_id = state_mapping["rhx"], state_mapping["rhy"]
    if separate_targets:
        rtx_id, rty_id = state_mapping["rtx"], state_mapping["rty"]
        ltx_id, lty_id = state_mapping["ltx"], state_mapping["lty"]
    else:
        ctx_id, cty_id = state_mapping["ctx"], state_mapping["cty"]

    if ax is None:
        fig = dv.AutoFigure("a")
        ax = fig.axes["a"]

    ax.scatter(
        x[:, rhx_id],
        x[:, rhy_id],
        color=wheel.grey,
        marker="x",
        s=10,
    )  # Actual hand position scattered
    ax.plot(x_post[:, rhx_id], x_post[:, rhy_id], color=wheel.pink)  # Posterior estiamte of the cursor
    if dual:
        lhx_id, lhy_id = state_mapping["lhx"], state_mapping["lhy"]
        ccx_id, ccy_id = state_mapping["ccx"], state_mapping["ccy"]
        ax.scatter(
            x[:, lhx_id],
            x[:, lhy_id],
            color=wheel.grey,
            marker="x",
            s=10,
        )  # Actual hand position scattered
        ax.plot(x_post[:, lhx_id], x_post[:, lhy_id], color=wheel.rak_blue)  # Posterior estiamte of the cursor

        ax.scatter(
            x[:, ccx_id],
            x[:, ccy_id],
            color=wheel.grey,
            marker="x",
            s=10,
        )  # Posterior estiamte of the center cursor
        ax.plot(
            x_post[:, ccx_id],
            x_post[:, ccy_id],
            color=wheel.sunflower,
        )  # Posterior estiamte of the center cursor

    if separate_targets:
        ax.scatter(x[-1, rtx_id], x[-1, rty_id], c=wheel.pink)
        ax.scatter(x[-1, ltx_id], x[-1, lty_id], facecolor="none", edgecolor=wheel.rak_blue)
    else:
        ax.scatter(x[-1, ctx_id], x[-1, cty_id], c=wheel.sunflower)

    ax.set_xlabel("X-Position")
    ax.set_ylabel("Y-Position")
    dv.Custom_Legend(
        ax,
        labels=["Right Hand", "Left Hand"],
        colors=[wheel.pink, wheel.rak_blue],
        fontsize=8,
    )

    dv.Custom_Legend(
        ax,
        labels=["P1", "P2"],
        colors=[wheel.rak_red, wheel.rak_blue],
        loc=(0.1, 1),
        fontsize=12,
    )
    if ax is None:
        return fig
    else:
        return ax


def plot_states_over_time(x, state_labels, state_mapping, linestyles, ax=None, colors=None):
    if ax is None:
        fig = dv.AutoFigure("a")
        ax = fig.axes["a"]
    if colors is None:
        colors = [
            wheel.pink,
            wheel.rak_blue,
            wheel.red,
            wheel.blue,
            wheel.sunflower,
            wheel.lighten_color(wheel.sunflower, 1.25),
        ]
    # Plot x,y traces for lh, rh, and cc
    state_ids = [state_mapping.get(k) for k in state_labels]
    for idx, label, ls, color in zip(state_ids, state_labels, linestyles, colors):
        ax.plot(x[:, idx], label=label, ls=ls, c=color)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("State")
    ax.legend()

    if ax is None:
        return fig
    else:
        return ax


def trace_plot_per_subject():
    """
    Should be a 1x2 plot, p1 on the left and p2 on the right
    Then each plot should have 4 traces, one for each condition
    """
    pass


def create_trajectory_animation(
    trial_type,
    rhx,
    rhy,
    lhx,
    lhy,
    ccx,
    ccy,
    rtx,
    rty,
    ltx,
    lty,
    p1_applied_force,
    p2_applied_force,
    timepoints,
    probe_timepoints,
    save_path=None,
    save_name=None,
    text_color="red",
    probe_onset=0,
    alpha=0.0,
    partner_knowledge=False,
    condition="self_small",
    pause_time=0,
    perturbation_type="Target Jump",
    target_width=0.04,
    target_height=0.01,
    save=False,
):
    """Creates an animation of model trajectories for a given condition

    Args:
        regular_df (pl.DataFrame): DataFrame containing model trajectories
        save_path (Path): Path to save the animation
        alpha (float): Model alpha parameter
        partner_knowledge (bool): Whether the model has knowledge of partner's target
        condition (str): Condition name to filter by
        trial_number (int): Trial number to animate
    """
    # Set up the figure
    if trial_type == "probe":
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(figsize=(4, 4))

    # Initialize line objects for trajectories (thin lines)
    (rh_line,) = ax1.plot([], [], "-", color=const.self_color, linewidth=2, alpha=1)
    (lh_line,) = ax1.plot([], [], "-", color=const.partner_color, linewidth=2, alpha=1)
    (cc_line,) = ax1.plot([], [], "-", color=const.cc_color, linewidth=2, alpha=1)

    # Initialize marker objects for current positions (larger circles)
    (rh_marker,) = ax1.plot([], [], "o", color=const.self_color, markersize=5, zorder=10)
    (lh_marker,) = ax1.plot([], [], "o", color=const.partner_color, markersize=5, zorder=10)
    (cc_marker,) = ax1.plot([], [], "o", color=const.cc_color, markersize=4, zorder=10)

    p1_target = mpl.patches.Rectangle(
        (rtx[0] - target_width / 2, rty[0] + target_height / 2),
        width=target_width,
        height=target_height,
        facecolor=const.self_color,
        edgecolor="none",
        lw=2,
        zorder=2,
        transform=ax1.transData,
    )
    p2_target = mpl.patches.Rectangle(
        (ltx[0] - target_width / 2, lty[0] + target_height / 2),
        width=target_width,
        height=target_height,
        facecolor="none",
        edgecolor=const.partner_color,
        lw=2,
        zorder=2,
        transform=ax1.transData,
    )
    # Create jump spot
    if trial_type != "regular":
        ax1.text(
            0.48,
            0.25 * (0.25) + rhy[0],
            "Cursor\nJump",
            ha="center",
            va="center",
            fontsize=6,
            color=wheel.grey,
            rotation=0,
        )
        ax1.axhline(
            0.25 * (0.25) + rhy[0],
            xmin=0.33,
            xmax=0.67,
            color="gray",
            linestyle="--",
            linewidth=0.8,
        )

    # Create vline and force channel arrows
    if trial_type == "probe":
        ax1.plot(
            (rhx[0], rhx[0]),
            (rhy[0], rhy[0] + 0.25),
            color="gray",
            linestyle="-",
            linewidth=1,
        )
        ax1.plot(
            (lhx[0], lhx[0]),
            (lhy[0], lhy[0] + 0.25),
            color="gray",
            linestyle="-",
            linewidth=1,
        )
        arrow_length = 0.03
        arrow_style = dict(
            head_width=0.005,
            head_length=0.005,
            color=wheel.white,
            lw=1.5,
            length_includes_head=True,
            alpha=1,
        )
        for ypos in np.linspace(0.07, 0.28, 6):
            # RIght hand arrows
            ax1.arrow(rhx[0] - arrow_length - 0.008, ypos, arrow_length, 0, **arrow_style)
            ax1.arrow(rhx[0] + arrow_length + 0.008, ypos, -arrow_length, 0, **arrow_style)
            # Left hand arrows
            ax1.arrow(lhx[0] - arrow_length - 0.008, ypos, arrow_length, 0, **arrow_style)
            ax1.arrow(lhx[0] + arrow_length + 0.008, ypos, -arrow_length, 0, **arrow_style)

        fps = 15
    else:
        fps = 45

    ax1.set_xlim(lhx[0] - 0.15, rhx[0] + 0.15)
    ax1.set_ylim(-0.01, 0.3)
    ax1.set_aspect("equal")
    ax1.set_axis_off()
    dv.Custom_Legend(
        ax1,
        colors=[const.partner_color, const.cc_color, const.self_color],
        labels=["Partner", "Center Cursor", "Self"],
        ncol=3,
        loc="upper center",
        columnspacing=4,
    )
    ax1.set_title(condition, fontsize=9, c=text_color)
    # ax1.set_position([-0.1, -0.02, 0.8, 0.8])

    # * Initialize ax2 for the feedback response
    if trial_type == "probe":
        ylim = (-4, 4)
        xticks = np.arange(0, 401, 100)

        (p1_force,) = ax2.plot([], [], "-", color=text_color, linewidth=2, alpha=1)
        (p2_force,) = ax2.plot([], [], "-", color=const.partner_color, linewidth=2, alpha=1)

        ax2.axvline(x=0, ls="--", c=wheel.grey, lw=0.75)
        ax2.text(
            0,
            ylim[1] / 2,
            perturbation_type,
            rotation=90,
            va="center",
            ha="right",
            color=wheel.grey,
            fontsize=6,
        )

        # Plot involuntary region
        # ax.fill_betweenx(np.arange(ylim[0],ylim[1],0.01), 180, 230, facecolor = wheel.lighten_color(wheel.light_grey,0.75), alpha = 0.1)
        ax2.text(
            205,
            ylim[-1],
            "Involuntary",
            rotation=90,
            va="top",
            ha="center",
            color=wheel.grey,
            fontsize=6,
        )
        ax2.axvline(x=180, color="grey", ls="-", lw=0.5, zorder=-10)
        ax2.axvline(x=230, color="grey", ls="-", lw=0.5, zorder=-10)

        # ax2.set_xticks(probe_timepoints)

        ax2.set_xlabel("Time From Cursor Jump (ms)")
        ax2.set_ylabel("Visuomotor Feedback Response [N]")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(ylim)
        # dv.Custom_Legend(ax2,
        #              colors=[const.partner_color, const.self_color],
        #              labels=["Partner", "Self"],ncol=1, loc="upper right", columnspacing=4)
        global index
        index = -1

    def init():
        # Initialize both lines and markers empty
        rh_line.set_data([], [])
        lh_line.set_data([], [])
        cc_line.set_data([], [])
        rh_marker.set_data([], [])
        lh_marker.set_data([], [])
        cc_marker.set_data([], [])
        ax1.add_patch(p1_target)
        ax1.add_patch(p2_target)
        if trial_type == "probe":
            p1_force.set_data([], [])
            p2_force.set_data([], [])
            return_tuple = (
                rh_line,
                lh_line,
                cc_line,
                rh_marker,
                lh_marker,
                cc_marker,
                p1_target,
                p2_target,
                p1_force,
            )
        else:
            return_tuple = (rh_line, lh_line, cc_line, rh_marker, lh_marker, cc_marker)
        return return_tuple

    def animate(frame):
        global index

        # Update trajectory lines (show full path up to current frame)
        rh_line.set_data(rhx[:frame], rhy[:frame])
        lh_line.set_data(lhx[:frame], lhy[:frame])
        cc_line.set_data(ccx[:frame], ccy[:frame])

        # Update current position markers (show only at current frame)
        rh_marker.set_data([rhx[frame]], [rhy[frame]])
        lh_marker.set_data([lhx[frame]], [lhy[frame]])
        cc_marker.set_data([ccx[frame]], [ccy[frame]])

        # Update Force applied
        if trial_type == "probe":
            if timepoints[frame] * 1000 >= probe_onset - 50:
                index += 1
                p1_force.set_data(probe_timepoints[:index], p1_applied_force[:index])
                p2_force.set_data(probe_timepoints[:index], p2_applied_force[:index])

        p1_target.set_xy((rtx[frame] - target_width / 2, rty[frame] + target_height / 2))
        p2_target.set_xy((ltx[frame] - target_width / 2, lty[frame] + target_height / 2))

        if trial_type == "probe":
            return_tuple = (
                rh_line,
                lh_line,
                cc_line,
                rh_marker,
                lh_marker,
                cc_marker,
                p1_force,
            )
        else:
            return_tuple = (rh_line, lh_line, cc_line, rh_marker, lh_marker, cc_marker)

        return return_tuple

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(rhx), interval=30, blit=True)

    # Save animation
    # NOTE Interval must be divisible by fps for ffmpeg to work properly, JK interval doesn't matter
    if save:
        anim.save(
            save_path / f"{save_name}_animation_alpha_{alpha}_pk_{partner_knowledge}_condition_{condition}.mp4",
            writer="ffmpeg",
            fps=fps,
        )
    # plt.close()


def plot_feedback_stats(
    ax,
    df,
    metric: str,
    xlocs: list[tuple[float, float]],
    yvals: list[float],
    condition_comparisons: list[tuple[str, str]],
    plot_cles=True,
    xloc_pad=0.1,
    **kwargs,
):
    if isinstance(yvals, (int, float)):
        yvals = [yvals]
    for j, comparison in enumerate(condition_comparisons):
        ttest_df = df.filter(
            (pl.col("metric") == metric)
            & ((pl.col("condition1") == comparison[0]) | (pl.col("condition2") == comparison[0]))
            & ((pl.col("condition1") == comparison[1]) | (pl.col("condition2") == comparison[1]))
        )
        if plot_cles:
            if ttest_df["cles"].item() < 50:
                cles = 100 - ttest_df["cles"].item()
            else:
                cles = ttest_df["cles"].item()
        else:
            cles = None
        dv.stat_annotation(
            ax,
            xlocs[j][0],
            xlocs[j][1],
            y=yvals[j],
            p_val=ttest_df["corrected_pval"].item(),
            cles=cles,
            **kwargs,
        )


def plot_towards_away_arrows(
    ax,
    arrowx,
    arrowys,
    dy,
    head_width,
    head_length,
    tail_width,
    text_yshifts: list[float],
    text_xshift: float = 19,
    transform=None,
    fontsize=6,
):
    if transform is None:
        transform = ax.transData

    ax.arrow(
        x=arrowx,
        y=arrowys[0],
        dx=0,
        dy=dy,
        head_width=head_width,
        head_length=head_length,
        color=wheel.grey,
        length_includes_head=True,
        transform=transform,
        shape="right",
        width=tail_width,
    )
    ax.text(
        arrowx + text_xshift,
        arrowys[0] + text_yshifts[0],
        "Towards\nTarget Jump",
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
        fontweight="bold",
        color=wheel.grey,
        transform=transform,
    )

    ax.arrow(
        x=arrowx,
        y=arrowys[1],
        dx=0,
        dy=-dy,
        head_width=head_width,
        head_length=head_length,
        color=wheel.grey,
        length_includes_head=True,
        transform=transform,
        shape="left",
        width=tail_width,
    )
    ax.text(
        arrowx + text_xshift,
        arrowys[1] + text_yshifts[1],
        "Away From\nTarget Jump",
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
        fontweight="bold",
        color=wheel.grey,
        transform=transform,
    )


def plot_reaching_traces(
    ax, 
    df_avg, 
    df_sd=None, 
    df_all=None, 
    alpha=0.2, 
    lw=0.5, 
    max_y=0.33, 
    mean_data_point_cutoff_index=-95,
    plot_left_hand=True,
):
    rhx = df_avg["filtered_right_handx"]
    rhy = np.where(df_avg["filtered_right_handy"] < max_y, df_avg["filtered_right_handy"], np.nan)
    lhx = -df_avg["filtered_left_handx"]
    lhy = np.where(df_avg["filtered_left_handy"] < max_y, df_avg["filtered_left_handy"], np.nan)
    ccx = df_avg["p1_center_cursor_pos_x"]
    ccy = np.where(df_avg["p1_center_cursor_pos_y"] < max_y, df_avg["p1_center_cursor_pos_y"], np.nan)

    ax.plot(rhx[:mean_data_point_cutoff_index], rhy[:mean_data_point_cutoff_index], color=const.self_color)
    if plot_left_hand:
        ax.plot(lhx[:mean_data_point_cutoff_index], lhy[:mean_data_point_cutoff_index], color=const.partner_color)
    ax.plot(ccx[:mean_data_point_cutoff_index], ccy[:mean_data_point_cutoff_index], color=const.cc_color)

    if df_sd is not None:
        rh_above_line = rhx + df_sd["filtered_right_handx"]
        rh_below_line = rhx - df_sd["filtered_right_handx"]
        lh_above_line = lhx + df_sd["filtered_left_handx"]
        lh_below_line = lhx - df_sd["filtered_left_handx"]
        cc_above_line = ccx + df_sd["p1_center_cursor_pos_x"]
        cc_below_line = ccx - df_sd["p1_center_cursor_pos_x"]
        # Extending the shaded area up a bit so mean doesn't poke out
        less_cutoff_rhy = np.where(
            df_avg["filtered_right_handy"] < max_y + 0.002, df_avg["filtered_right_handy"], np.nan
        )
        less_cutoff_lhy = np.where(df_avg["filtered_left_handy"] < max_y + 0.002, df_avg["filtered_left_handy"], np.nan)
        less_cutoff_ccy = np.where(
            df_avg["p1_center_cursor_pos_y"] < max_y + 0.002, df_avg["p1_center_cursor_pos_y"], np.nan
        )
        ax.fill_betweenx(
            less_cutoff_rhy,
            rh_above_line,
            rh_below_line,
            alpha=alpha,
            color=const.self_color,
            edgecolor="none",
        )
        if plot_left_hand:
            ax.fill_betweenx(
                less_cutoff_lhy,
                lh_above_line,
                lh_below_line,
                alpha=alpha,
                color=const.partner_color,
                edgecolor="none",
            )
        ax.fill_betweenx(
            less_cutoff_ccy,
            cc_above_line,
            cc_below_line,
            alpha=alpha,
            color=const.cc_color,
            edgecolor="none",
        )
    elif df_all is not None:
        traces = df_all.group_by(pl.col("experiment_trial_number"), maintain_order=True).agg(
            pl.col(
                "filtered_right_handx",
                "filtered_right_handy",
                "filtered_left_handx",
                "filtered_left_handy",
                "p1_center_cursor_pos_x",
                "p1_center_cursor_pos_y",
            )
        )
        rhx = np.vstack(traces["filtered_right_handx"]).T
        rhy = np.vstack(traces["filtered_right_handy"]).T
        lhx = np.vstack(traces["filtered_left_handx"]).T
        lhy = np.vstack(traces["filtered_left_handy"]).T
        ccx = np.vstack(traces["p1_center_cursor_pos_x"]).T
        ccy = np.vstack(traces["p1_center_cursor_pos_y"]).T
        # Get first point that they crossed max_y for each trial
        # Need to do this bc people sometimes didn't hold in final target, then came back
        # so i don't want to plot that
        rhy_first_idx = np.argmax(rhy > max_y, axis=0)
        lhy_first_idx = np.argmax(lhy > max_y, axis=0)
        ccy_first_idx = np.argmax(ccy > max_y, axis=0)
        # Loop through and plot to use the idx to set nans
        for i in range(rhy.shape[1]):
            rhy[rhy_first_idx[i] :, i] = np.nan
            lhy[lhy_first_idx[i] :, i] = np.nan
            ccy[ccy_first_idx[i] :, i] = np.nan

            ax.plot(rhx[:, i], rhy[:, i], color=const.self_color, lw=lw, alpha=alpha)
            if plot_left_hand:
                ax.plot(-lhx[:, i], lhy[:, i], color=const.partner_color, lw=lw, alpha=alpha)
            ax.plot(
                ccx[:, i],
                ccy[:, i],
                color=const.cc_color,
                lw=lw,
                alpha=alpha,
            )
    ax.set_xlabel("Lateral Position (m)", fontsize=6.5)
    # ax.yaxis.set_label_coords(-0.15)
    ax.set_ylabel("Forward Position (m)", fontsize=6.5)
    xmin, xmax = 0.54 - 0.18, 0.54 + 0.18
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([0.44, 0.54, 0.64], [-0.1, 0, 0.1])

    ymin, ymax = 0.03, 0.38
    ax.set_yticks([0.08, 0.18, 0.28, 0.38], [0, 0.1, 0.2, 0.3])
    ax.set_ylim(ymin, ymax)
    if plot_left_hand:
        ax.text(
            0.44,
            0.075,
            "Partner\nCursor",
            fontsize=5.5,
            fontweight="bold",
            color=const.partner_color,
            transform=ax.transData,
            ha="center",
            va="top",
        )
    ax.text(
        0.54,
        0.075,
        "Center\nCursor",
        fontsize=5.5,
        fontweight="bold",
        color=const.cc_color,
        transform=ax.transData,
        ha="center",
        va="top",
    )
    ax.text(
        0.64,
        0.075,
        "Self\nCursor",
        fontsize=5.5,
        fontweight="bold",
        color=const.self_color,
        transform=ax.transData,
        ha="center",
        va="top",
    )

    ax.tick_params(bottom=True, left=True, pad=1, width=1)


def plot_posture_traces(ax, df_avg, df_sd=None, df_all=None, alpha=0.2, lw=0.5, plot_left_hand=True):
    ax.plot(df_avg["filtered_right_handx"], color=const.self_color)
    if plot_left_hand:
        ax.plot(-df_avg["filtered_left_handx"], color=const.partner_color)
    ax.plot(df_avg["p1_center_cursor_pos_x"], color=const.cc_color)
    # ax.plot(df_avg['p1_self_target_pos_x'], color=const.self_color, ls='-', lw=2.5, zorder=-99, alpha=0.8)
    # ax.plot(df_avg['p1_partner_target_pos_x'], color=const.partner_color, ls='--', lw=2.5, zorder=-99, alpha=0.8)
    if df_sd is not None:
        rh_above_line = df_avg["filtered_right_handx"] + df_sd["filtered_right_handx"]
        rh_below_line = df_avg["filtered_right_handx"] - df_sd["filtered_right_handx"]
        lh_above_line = -(df_avg["filtered_left_handx"] + df_sd["filtered_left_handx"])
        lh_below_line = -(df_avg["filtered_left_handx"] - df_sd["filtered_left_handx"])
        cc_above_line = df_avg["p1_center_cursor_pos_x"] + df_sd["p1_center_cursor_pos_x"]
        cc_below_line = df_avg["p1_center_cursor_pos_x"] - df_sd["p1_center_cursor_pos_x"]
        ax.fill_between(
            df_avg["timepoint"],
            rh_above_line,
            rh_below_line,
            alpha=alpha,
            color=const.self_color,
            edgecolor="none",
        )
        if plot_left_hand:
            ax.fill_between(
                df_avg["timepoint"],
                lh_above_line,
                lh_below_line,
                alpha=alpha,
                color=const.partner_color,
                edgecolor="none",
            )
        ax.fill_between(
            df_avg["timepoint"],
            cc_above_line,
            cc_below_line,
            alpha=alpha,
            color=const.cc_color,
            edgecolor="none",
        )

    elif df_all is not None:
        traces = df_all.group_by(pl.col("experiment_trial_number"), maintain_order=True).agg(
            pl.col(
                "filtered_right_handx",
                "filtered_left_handx",
                "p1_center_cursor_pos_x",
                "timepoint",
            )
        )
        rhx = np.vstack(traces["filtered_right_handx"]).T
        lhx = np.vstack(traces["filtered_left_handx"]).T
        ccx = np.vstack(traces["p1_center_cursor_pos_x"]).T
        timepoints = np.vstack(traces["timepoint"])[0]
        ax.plot(timepoints, rhx, color=const.self_color, lw=lw, alpha=alpha)
        if plot_left_hand:
            ax.plot(timepoints, -lhx, color=const.partner_color, lw=lw, alpha=alpha)
        ax.plot(timepoints, ccx, color=const.cc_color, lw=lw, alpha=alpha)
    # ax.text(650, 0.01,"Target Hit", va='bottom', ha='right', fontsize=5.5, c=wheel.rak_blue, fontweight='bold',
    #            transform= transforms.blended_transform_factory(ax.transData, ax.transAxes),
    # )
    # ax.scatter(667, 0, marker='|', s=100,
    #            transform= transforms.blended_transform_factory(ax.transData, ax.transAxes),
    #            clip_on=True,
    # )
    ymin, ymax = 0.54 + 0.18, 0.54 - 0.18
    ax.set_yticks([0.44, 0.54, 0.64], ["-0.1", 0, 0.1])
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.linspace(0, 800, 5))  # , labels=[0,100])
    ax.set_xlim(0, 667)
    ax.set_xlabel("Time (ms)", fontsize=6.5, labelpad=0)
    ax.set_ylabel("Lateral Position (m)", fontsize=6.5, labelpad=0.5)

    if plot_left_hand:
        ax.text(
            0.01,
            0.62,
            "Partner\nCursor",
            fontsize=5.5,
            fontweight="bold",
            color=const.partner_color,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )
    ax.text(
        0.01,
        0.32,
        "Center\nCursor",
        fontsize=5.5,
        fontweight="bold",
        color=const.cc_color,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.01,
        0.057,
        "Self\nCursor",
        fontsize=5.5,
        fontweight="bold",
        color=const.self_color,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )
    # ax.tick_params(bottom=False,left=False,  pad=0)


def plot_color_means(ax, xlocs, means, colors, edgecolor=None, size=25):
    if edgecolor is None:
        edgecolor = wheel.lighten_color(wheel.dark_grey, 1.2)
    ax.scatter(
        xlocs,
        means,
        facecolors=colors,
        edgecolors=edgecolor,
        zorder=1000,
        s=size,
    )
    ax.plot(
        xlocs,
        means,
        color=edgecolor,
        zorder=999,
    )


def plot_jump_targets(
    ax,
    x,
    y,
    jump_dist,
    width,
    height,
    target_arrow_length,
    probe,
    target_arrow_label,
    plot_arrows,
    alpha=1,
    partner_jump_dist=0,
    transform=None,
):
    if transform is None:
        transform = ax.transAxes
    self_patch = patches.Rectangle(
        ((x - width / 2) + jump_dist, y + height / 2),
        width=width,
        height=height,
        facecolor=const.self_color,
        edgecolor=const.self_color,
        lw=1,
        zorder=2,
        alpha=alpha,
        transform=transform,
        clip_on=False,
    )
    ax.add_patch(self_patch)

    partner_patch = patches.Rectangle(
        ((x - width / 2) + partner_jump_dist, y + height / 2),
        width=width,
        height=height,
        facecolor="none",
        edgecolor=const.partner_color,
        lw=1,
        zorder=2,
        transform=transform,
        ls="-",
        alpha=alpha,
        clip_on=False,
    )
    ax.add_patch(partner_patch)

    if plot_arrows:
        partner_jump_arrow_yshift = 0
        if jump_dist != 0:
            plot_jump_arrows(
                ax,
                x=self_patch.get_x(),
                y=self_patch.get_y() - 0.05,
                dx=width,
                dy=0,
                arrow_yshift=0.01,
                condition=const.condition_names[1],
                arrow_head_width=0.05,
                arrow_head_length=0.03,
                arrow_lw=1,
                transform=transform,
                tail_width=0.015,
            )
            partner_jump_arrow_yshift = -0.01

        if partner_jump_dist != 0:
            plot_jump_arrows(
                ax,
                x=partner_patch.get_x(),
                y=partner_patch.get_y() - 0.05 - partner_jump_arrow_yshift,
                dx=width,
                dy=0,
                arrow_yshift=0.01,
                condition=const.condition_names[2],
                arrow_head_width=0.05,
                arrow_head_length=0.03,
                arrow_lw=1,
                transform=transform,
                tail_width=0.015,
            )

        # ax.text(0.5+0.48*target_arrow_length, ymax-0.08-0.01, target_arrow_label, ha='center', va='top', color=wheel.grey, fontweight='bold', fontsize=6.5 ) # T1
        if probe:
            plot_jump_arrows(
                ax,
                x=self_patch.get_x() + width,
                y=self_patch.get_y() + 0.04 + height,
                dx=-width,
                dy=0,
                arrow_yshift=0.01,
                condition=const.condition_names[1],
                arrow_head_width=0.05,
                arrow_head_length=0.03,
                arrow_lw=1,
                transform=ax.transAxes,
                tail_width=0.015,
            )
            # ax.arrow(0.5+target_arrow_length, ymax+0.026, -target_arrow_length, 0, **jump_arrow_style, shape='left',)
            # ax.text(0.5+0.55*target_arrow_length, ymax+0.028, "2", ha='center', va='bottom', color=wheel.grey, fontweight='bold', fontsize=6.5 ) # T2


def plot_jump_targets_posture_traces(
    ax,
    x,
    y,
    jump_dist,
    width,
    height,
    target_arrow_length,
    probe,
    target_arrow_label,
    plot_arrows,
    alpha=1,
    partner_jump_dist=0,
    transform=None,
):
    if transform is None:
        transform = ax.transAxes
    self_patch = patches.Rectangle(
        ((x - width / 2), y + height / 2 + jump_dist),
        width=width,
        height=height,
        facecolor=const.self_color,
        edgecolor=const.self_color,
        lw=1,
        zorder=2,
        alpha=alpha,
        transform=transform,
        clip_on=False,
    )
    ax.add_patch(self_patch)

    partner_patch = patches.Rectangle(
        ((x - width / 2), y + height / 2 + partner_jump_dist),
        width=width,
        height=height,
        facecolor="none",
        edgecolor=const.partner_color,
        lw=1,
        zorder=2,
        transform=transform,
        ls="-",
        alpha=alpha,
        clip_on=False,
    )
    ax.add_patch(partner_patch)
