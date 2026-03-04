import numpy as np
import polars as pl
import src.constants as const
import warnings


class Dyad:
    def __init__(
        self,
        experiment: str,
        data_type: str,
        raw_df: pl.DataFrame | None,
        filt_df: pl.DataFrame | None,
        norm_df: pl.DataFrame,
        probe_df: pl.DataFrame,
        normalized_probe_df: pl.DataFrame | None = None,
        auc_df: pl.DataFrame | None = None,
        extrapolation_onset_df: pl.DataFrame | None = None,
        trial_df: pl.DataFrame | None = None,
        trial_feedback_response_df: pl.DataFrame | None = None,
        condition_names_collapsed: list[str] | None = None,
    ):
        """
        Initializes the Dyad class.

        Parameters
        ----------
        experiment : str
            Exp1 or Exp2.

        data_type : str
            individual or group

        raw_df : DataFrame
            Raw data frame.

        filt_df : DataFrame
            Filtered data frame.

        norm_df : DataFrame
            Normalized data frame.

        probe_df : DataFrame
            Probe trial data frame.
        """
        self.raw_df = raw_df
        self.filt_df = filt_df
        self.norm_df = norm_df
        self.probe_df = probe_df
        self.normalized_probe_df = normalized_probe_df
        self.auc_df = auc_df
        self.extrapolation_onset_df = extrapolation_onset_df
        self.trial_df = trial_df
        self.trial_feedback_response_df = trial_feedback_response_df
        self.experiment = experiment
        self.data_type = data_type
        self.condition_names_collapsed = (
            condition_names_collapsed if condition_names_collapsed is not None else const.condition_names_collapsed
        )

        try:
            self.NUM_SUBJECTS = probe_df["subject"].max()
        except:
            self.NUM_SUBJECTS = norm_df["subject"].max()

        # * Average and sd across subjects
        # * (who have already been averaged over condition, perturbation_type, and timepoint)
        self.norm_cols = (
            "filtered_right_fs_forcex",
            "filtered_left_fs_forcex",
            "filtered_right_handx",
            "filtered_left_handx",
            "filtered_right_handy",
            "filtered_left_handy",
            "p1_center_cursor_pos_x",
            "p1_center_cursor_pos_y",
            "p2_center_cursor_pos_x",
            "p2_center_cursor_pos_y",
            "p1_self_target_pos_x",
            "p1_partner_target_pos_x",
            # "p1_self_target_pos_y", "p1_partner_target_pos_x"
        )
        self.probe_cols = ("filtered_right_fs_forcex", "filtered_left_fs_forcex")

    @property
    def avg_norm_df(self):
        return self.norm_df.group_by(["perturbation_type", "condition", "timepoint"], maintain_order=True).agg(
            pl.col(self.norm_cols).mean()
        )

    @property
    def sd_norm_df(self):
        return self.norm_df.group_by(["perturbation_type", "condition", "timepoint"], maintain_order=True).agg(
            pl.col(self.norm_cols).std()
        )

    @property
    def avg_probe_df(self):
        return self.probe_df.group_by(["perturbation_type", "condition", "timepoint"], maintain_order=True).agg(
            pl.col(self.probe_cols).mean()
        )

    @property
    def sd_probe_df(self):
        return self.probe_df.group_by(["perturbation_type", "condition", "timepoint"], maintain_order=True).agg(
            pl.col(self.probe_cols).std()
        )

    @property
    def collapsed_probe_df(self):
        return self.get_collapsed_df(
            self.probe_df,
            p1_metric_names=["filtered_right_fs_forcex"],
            p2_metric_names=["filtered_left_fs_forcex"],
            new_metric_names=["force"],
        )

    @property
    def collapsed_norm_df(self):
        return self.get_collapsed_df(
            self.norm_df,
            p1_metric_names=[
                "filtered_right_handx",
                "filtered_right_handy",
                "filtered_right_handxvel",
                "filtered_right_handyvel",
            ],
            p2_metric_names=[
                "filtered_left_handx",
                "filtered_left_handy",
                "filtered_left_handxvel",
                "filtered_left_handyvel",
            ],
            new_metric_names=["xpos", "ypos", "xvel", "yvel"],
        )

    @property
    def collapsed_trial_df(self):
        return self.get_collapsed_df(
            self.trial_df,
            p1_metric_names=["p1_movement_onset_time", "p1_lateral_pos_at_cc_hit_time"],
            p2_metric_names=["p2_movement_onset_time", "p2_lateral_pos_at_cc_hit_time"],
            new_metric_names=["movement_onset_time", "lateral_pos_at_cc_hit_time"],
            include_timepoints=False,
        )

    def _get_columns(self, metric):
        columns = []
        if metric in ["pos", "position"]:
            columns = [
                "filtered_right_handx",
                "filtered_right_handy",
                "filtered_left_handx",
                "filtered_left_handy",
                "p1_center_cursor_pos_x",
                "p1_center_cursor_pos_y",
                "p2_center_cursor_pos_x",
                "p2_center_cursor_pos_y",
            ]
        if metric in ["vel", "velocity"]:
            columns = [
                "filtered_right_handxvel",
                "filtered_right_handyvel",
                "filtered_left_handxvel",
                "filtered_left_handyvel",
            ]
        if metric in ["f", "force"]:
            columns = ["filtered_right_fs_forcex", "filtered_left_fs_forcex"]
        return columns + [
            "Experiment Trial Number",
            "perturbation_type",
            "condition",
            "subject",
        ]

    def get_group_avg_df(
        self,
        col_names,
        df_type: str,
        collapsed: str,
        summary_stat: str,
    ):
        if df_type == "norm":
            if collapsed:
                df = self.collapsed_norm_df
            else:
                df = self.norm_df
        elif df_type == "probe":
            if collapsed:
                df = self.collapsed_probe_df
            else:
                df = self.probe_df
        else:
            raise ValueError("df_type must be 'norm' or 'probe'")

        dff = df.group_by(["perturbation_type", "condition", "timepoint"], maintain_order=True)
        if summary_stat == "mean":
            dff = dff.agg(pl.col(col_names).mean())
        elif summary_stat == "std":
            dff = dff.agg(pl.col(col_names).std())
        else:
            raise ValueError("'summary_stat' should be either 'mean' or 'std'")

        return dff

    def get_collapsed_df(
        self,
        df,
        p1_metric_names,
        p2_metric_names,
        new_metric_names,
        filters: list[pl.Expr] | None = None,
        p1_filters: list[pl.Expr] | None = None,
        p2_filters: list[pl.Expr] | None = None,
        include_timepoints: bool = True,
        include_mini_block_number: bool = False,
    ):
        assert len(p1_metric_names) == len(p2_metric_names) == len(new_metric_names)
        if filters is not None:
            p1_filters = filters
            p2_filters = filters

        all_collapsed_data = {k: [] for k in new_metric_names}
        collapsed_conditions = []
        subjects = []
        perturbation_types = []
        timepoints = []
        mini_block_numbers = []

        for i, condition in enumerate(self.condition_names_collapsed):
            p1_jump_type = const.collapsed_conditions_to_p1_p2[condition][
                "p1"
            ]  # Maps the collapsed condition name to the jump_type for filtering of the og dfs
            p2_jump_type = const.collapsed_conditions_to_p1_p2[condition]["p2"]

            p1_df = df.filter(
                pl.col("condition") == p1_jump_type,
            )
            p2_df = df.filter(
                pl.col("condition") == p2_jump_type,
            )
            if p1_filters is not None and p2_filters is not None:
                p1_df = p1_df.filter(p1_filters)
                p2_df = p2_df.filter(p2_filters)

            for j, (p1_metric_name, p2_metric_name) in enumerate(zip(p1_metric_names, p2_metric_names)):
                all_collapsed_data[new_metric_names[j]].append(
                    np.hstack((p1_df[p1_metric_name], p2_df[p2_metric_name]))
                )

            # Create identifiers for long format
            subs = np.hstack(
                (
                    np.char.add(p1_df["subject"].to_numpy().astype(str), "R"),
                    np.char.add(p2_df["subject"].to_numpy().astype(str), "L"),
                )
            )
            perturbation_type = np.hstack(
                (
                    p1_df["perturbation_type"].to_numpy(),
                    p2_df["perturbation_type"].to_numpy(),
                )
            )
            # display(p1_df)
            subjects.append(subs)
            # Want this condition repeated for num_timepoints*subject*2
            collapsed_conditions.append([condition] * len(subs))
            if include_timepoints:
                timepoints.append(np.tile(p1_df["timepoint"], 2))
            else:
                timepoints.append(np.tile([0] * len(p1_df), 2))
            perturbation_types.append(perturbation_type)
            
            if include_mini_block_number:
                mini_block_numbers.append(np.tile(p1_df["mini_block_number"], 2))

        collapsed_df = pl.DataFrame()
        collapsed_df = collapsed_df.with_columns(
            subject=np.array(subjects).flatten(),
            timepoint=np.array(timepoints).flatten(),
            condition=np.array(collapsed_conditions).flatten(),
            perturbation_type=np.array(perturbation_types).flatten(),
        )
        if include_mini_block_number:
            collapsed_df = collapsed_df.with_columns(
                mini_block_number=np.array(mini_block_numbers).flatten(),
            )
        final_data = {k: np.array(v).flatten() for k, v in all_collapsed_data.items()}
        collapsed_df = collapsed_df.with_columns(**final_data)
            
        # new_columns = {}
        # for data,new_metric_name in all_collapsed_data,new_metric_names:
        #     new_columns[new_metric_name] = np.array(data[new_metric_name]).flatten()

        # collapsed_df = collapsed_df.with_columns(**new_columns)
        return collapsed_df

    def groupby_to_numpy(self, metric, df="norm"):
        """
        metric:
        df: 'raw', 'norm', 'probe'
        """
        # TODO figure out how I want to do this
        if df == "norm":
            df = self.norm_df
        elif df == "filt":
            df = self.filt_df
        elif df == "probe":
            df = self.probe_df
        else:
            raise ValueError("df must be 'norm', 'filt', or 'probe'")
        return np.vstack(
            df.group_by("Experiment Trial Number", maintain_order=True).agg(pl.col(metric))[metric].to_numpy()
        )

    def get_traces(
        self,
        metric,
        condition,
        perturbation_type,
        average_timepoints=False,
        cursor=None,
        df_type="norm",
        **kwargs,
    ):
        """
        cursor: Str -> None, "p1", "p2", "center"
        condition: Str -> "p1_jump", "p2_jump", "same_jump", "opposite_jump"
        perturbation_type: Str ->
        """
        group_by_cols = kwargs.get("group_by_cols", ["subject", "timepoint", "perturbation_type", "condition"])
        columns = self._get_columns(metric)
        if df_type == "norm":
            df = self.norm_df  # Still filtered
        elif df_type == "filt":
            df = self.filt_df
        elif df_type == "probe":
            df = self.probe_df  # Still filtered, but just takes 400ms after the probe
        else:
            raise ValueError("df_type must be 'norm', 'filt', 'probe'")

        if average_timepoints:
            df = df.group_by(group_by_cols, maintain_order=True).mean()

        dff = df.filter((pl.col("perturbation_type") == perturbation_type) & (pl.col("condition") == condition))

        if cursor is None:
            return dff
        elif cursor == "p1":
            return dff.select(pl.col("^Filtered_Right_.*$"))
        elif cursor == "p2":
            return dff.select(pl.col("^Filtered_Left_.*$"))
        elif cursor == "center":
            return dff.select(pl.col("^_center_cursor_.*$"))
        else:
            raise ValueError("cursor must be p1, p2, center, or None")

    def get_force_mean_df(self, timepoint1: int, timepoint2: int, collapsed: bool):
        if collapsed:
            df = self.get_right_minus_left_probe_df(collapsed=collapsed)
            cols = ["normalized_force"]
        else:
            df = self.get_right_minus_left_probe_df(collapsed=collapsed)
            cols = ["p1_normalized_force", "p2_normalized_force"]

        dff = df.filter(pl.col("timepoint").is_between(timepoint1, timepoint2, closed="both"))

        mean_force_df = dff.group_by(["subject", "condition"], maintain_order=True).agg(pl.col(cols).mean())

        return mean_force_df

    def get_right_minus_left_probe_df(self, collapsed, df=None, divide_by_two=True):
        if divide_by_two:
            div = 2
        else:
            div = 1
        if collapsed:
            if df is None:
                df = self.collapsed_probe_df
            p1_right_data = df.filter(
                pl.col("perturbation_type") == "probe_right",
                pl.col("subject").str.contains("R"),
            )
            p1_left_data = df.filter(
                pl.col("perturbation_type") == "probe_left",
                pl.col("subject").str.contains("R"),
            )
            p2_right_data = df.filter(
                pl.col("perturbation_type") == "probe_right",
                pl.col("subject").str.contains("L"),
            )
            p2_left_data = df.filter(
                pl.col("perturbation_type") == "probe_left",
                pl.col("subject").str.contains("L"),
            )
            new_df = pl.DataFrame(
                {
                    "timepoint": np.array([p1_right_data["timepoint"], p2_right_data["timepoint"]]).flatten(),
                    "subject": np.array([p1_right_data["subject"], p2_right_data["subject"]]).flatten(),
                    "condition": np.array([p1_right_data["condition"], p2_right_data["condition"]]).flatten(),
                    "normalized_force": np.array(
                        [
                            (p1_right_data["force"] - p1_left_data["force"]) / div,
                            (p2_left_data["force"] - p2_right_data["force"]) / div,
                        ]
                    ).flatten(),
                }
            )
            if "mini_block_number" in df.columns:
                new_df = new_df.with_columns(
                    mini_block_number=np.array([p1_right_data["mini_block_number"], p2_right_data["mini_block_number"]]).flatten(),
                )
        else:
            if df is None:
                df = self.probe_df
            right_data = df.filter(pl.col("perturbation_type") == "probe_right")
            left_data = df.filter(pl.col("perturbation_type") == "probe_left")
            # Normalize
            p1_right = right_data["filtered_right_fs_forcex"]
            p1_left = left_data["filtered_right_fs_forcex"]
            p2_right = right_data["filtered_left_fs_forcex"]
            p2_left = left_data["filtered_left_fs_forcex"]
            # Piece back together into new dataframe
            p1 = (p1_right - p1_left) / div
            p2 = (p2_left - p2_right) / div
            new_df = pl.DataFrame(
                {
                    "timepoint": np.array([right_data["timepoint"]]).flatten(),
                    "subject": np.array([right_data["subject"]]).flatten(),
                    "condition": np.array([right_data["condition"]]).flatten(),
                    "p1_normalized_force": np.array([p1]).flatten(),
                    "p2_normalized_force": np.array([p2]).flatten(),
                }
            )
        return new_df

    def get_neutral_normalized_probe_df(self, combine_left_right, collapsed=False):
        if collapsed:
            df = self.collapsed_probe_df
            # ! Separating by subject R and L because p1 needs right - left and p2 needs left - right
            p1_neutral_data = df.filter(
                pl.col("perturbation_type") == "probe_neutral",
                pl.col("subject").str.contains("R"),
            )
            p1_right_data = df.filter(
                pl.col("perturbation_type") == "probe_right",
                pl.col("subject").str.contains("R"),
            )
            p1_left_data = df.filter(
                pl.col("perturbation_type") == "probe_left",
                pl.col("subject").str.contains("R"),
            )
            p2_neutral_data = df.filter(
                pl.col("perturbation_type") == "probe_neutral",
                pl.col("subject").str.contains("L"),
            )
            p2_right_data = df.filter(
                pl.col("perturbation_type") == "probe_right",
                pl.col("subject").str.contains("L"),
            )
            p2_left_data = df.filter(
                pl.col("perturbation_type") == "probe_left",
                pl.col("subject").str.contains("L"),
            )

            p1_norm_right = p1_right_data["force"] - p1_neutral_data["force"]
            p1_norm_left = p1_left_data["force"] - p1_neutral_data["force"]
            p2_norm_right = p2_right_data["force"] - p2_neutral_data["force"]
            p2_norm_left = p2_left_data["force"] - p2_neutral_data["force"]
            # Piece back together into new dataframe
            if combine_left_right:
                p1_norm = p1_norm_right - p1_norm_left
                p2_norm = p2_norm_left - p2_norm_right
                new_df = pl.DataFrame(
                    {
                        "timepoint": np.array([p1_right_data["timepoint"], p2_right_data["timepoint"]]).flatten(),
                        "subject": np.array([p1_right_data["subject"], p2_right_data["subject"]]).flatten(),
                        "condition": np.array([p1_right_data["condition"], p2_right_data["condition"]]).flatten(),
                        "normalized_force": np.array([p1_norm, p2_norm]).flatten(),
                    }
                )
            else:
                new_df = pl.DataFrame(
                    {
                        "timepoint": np.array(
                            [
                                p1_right_data["timepoint"],
                                p2_right_data["timepoint"],
                                p1_left_data["timepoint"],
                                p2_left_data["timepoint"],
                            ]
                        ).flatten(),
                        "subject": np.array(
                            [
                                p1_right_data["subject"],
                                p2_right_data["subject"],
                                p1_left_data["subject"],
                                p2_left_data["subject"],
                            ]
                        ).flatten(),
                        "perturbation_type": np.array(
                            [
                                p1_right_data["perturbation_type"],
                                p2_right_data["perturbation_type"],
                                p1_left_data["perturbation_type"],
                                p2_left_data["perturbation_type"],
                            ]
                        ).flatten(),
                        "condition": np.array(
                            [
                                p1_right_data["condition"],
                                p2_right_data["condition"],
                                p1_left_data["condition"],
                                p2_left_data["condition"],
                            ]
                        ).flatten(),
                        "normalized_force": np.array(
                            [p1_norm_right, p2_norm_right, p1_norm_left, p2_norm_left]
                        ).flatten(),
                    }
                )

        else:
            df = self.probe_df
            neutral_data = df.filter(pl.col("perturbation_type") == "probe_neutral")
            right_data = df.filter(pl.col("perturbation_type") == "probe_right")
            left_data = df.filter(pl.col("perturbation_type") == "probe_left")
            # Normalize
            p1_norm_right = right_data["filtered_right_fs_forcex"] - neutral_data["filtered_right_fs_forcex"]
            p1_norm_left = left_data["filtered_right_fs_forcex"] - neutral_data["filtered_right_fs_forcex"]
            p2_norm_right = right_data["filtered_left_fs_forcex"] - neutral_data["filtered_left_fs_forcex"]
            p2_norm_left = left_data["filtered_left_fs_forcex"] - neutral_data["filtered_left_fs_forcex"]
            # Piece back together into new dataframe
            if combine_left_right:
                p1_norm = p1_norm_right - p1_norm_left
                p2_norm = p2_norm_left - p2_norm_right
                new_df = pl.DataFrame(
                    {
                        "timepoint": np.array([right_data["timepoint"]]).flatten(),
                        "subject": np.array([right_data["subject"]]).flatten(),
                        "condition": np.array([right_data["condition"]]).flatten(),
                        "p1_normalized_force": np.array([p1_norm]).flatten(),
                        "p2_normalized_force": np.array([p2_norm]).flatten(),
                    }
                )
            else:
                new_df = pl.DataFrame(
                    {
                        "timepoint": np.array([right_data["timepoint"], left_data["timepoint"]]).flatten(),
                        "subject": np.array([right_data["subject"], left_data["subject"]]).flatten(),
                        "perturbation_type": np.array(
                            [
                                right_data["perturbation_type"],
                                left_data["perturbation_type"],
                            ]
                        ).flatten(),
                        "condition": np.array([right_data["condition"], left_data["condition"]]).flatten(),
                        "p1_normalized_force": np.array([p1_norm_right, p1_norm_left]).flatten(),
                        "p2_normalized_force": np.array([p2_norm_right, p2_norm_left]).flatten(),
                    }
                )
            # display(new_df)
        return new_df

    @property
    def involuntary_force_mean_df(self):
        return self.get_force_mean_df(
            timepoint1=180,
            timepoint2=230,
            collapsed=True,
        )

    @property
    def semi_involuntary_force_mean_df(self):
        return self.get_force_mean_df(
            timepoint1=230,
            timepoint2=300,
            collapsed=True,
        )

    @property
    def voluntary_force_mean_df(self):
        return self.get_force_mean_df(
            timepoint1=300,
            timepoint2=400,
            collapsed=True,
        )

    @property
    def full_voluntary_force_mean_df(self):
        return self.get_force_mean_df(
            timepoint1=400,
            timepoint2=600,
            collapsed=True,
        )

    def get_collapsed_trial_feedback_response_df(self, response_region="involuntary", 
                                                 trials:str="all", left_minus_right:bool=True): 
        # filter to involuntary only
        df = self.trial_feedback_response_df.filter(
            ~pl.col("perturbation_type").str.contains("probe_neutral"),
            pl.col("response_type") == response_region    
        )
        # Collapse conditions
        df = self.get_collapsed_df(df, 
                            p1_metric_names=["filtered_right_fs_forcex"], 
                            p2_metric_names=["filtered_left_fs_forcex"], 
                            new_metric_names=["force"], 
                            include_mini_block_number=True,
                            include_timepoints=False,
        )                     
        if trials == "first_half":
            df = df.filter(pl.col("mini_block_number") <=5)
        elif trials == "second_half":
            df = df.filter(pl.col("mini_block_number") > 5)
        
        # Average across left and right probes 
        if left_minus_right:
            df = self.get_right_minus_left_probe_df(collapsed=True, df=df, divide_by_two=True)

        return df 

    def add_p1_p2_target_type(self, df):
        df = df.with_columns(
            p1_target_type=pl.when(pl.col("target_type").is_in(["p1_relevant", "joint_relevant"]))
            .then(pl.lit("relevant"))
            .otherwise(pl.lit("irrelevant")),
            p2_target_type=pl.when(pl.col("target_type").is_in(["p2_relevant", "joint_relevant"]))
            .then(pl.lit("relevant"))
            .otherwise(pl.lit("irrelevant")),
        )
        return df

    def get_peak_velocity_df(
        self,
        direction="forward",
        collapsed=True,
    ):
        if direction == "forward":
            p1_metric = "filtered_right_handyvel"
            p2_metric = "filtered_left_handyvel"
            new_metric_name = "forward_velocity"
        elif direction == "lateral":
            p1_metric = "filtered_right_handxvel"
            p2_metric = "filtered_left_handxvel"
            new_metric_name = "lateral_velocity"
        else:
            raise ValueError("direction must be 'forward' or 'lateral'")
        if collapsed:
            collapsed_df = (
                self.get_collapsed_df(
                    self.norm_df,
                    p1_metric_names=[p1_metric],
                    p2_metric_names=[p2_metric],
                    new_metric_names=[new_metric_name],
                )
                .group_by(pl.col("subject", "condition", "timepoint"), maintain_order=True)
                .agg(pl.col(new_metric_name).max().alias("peak_" + new_metric_name))
            )
            return collapsed_df
        else:
            return self.norm_df.group_by(pl.col("subject", "condition", "timepoint"), maintain_order=True).agg(
                pl.col(p1_metric).max().alias("peak_" + p1_metric),
                pl.col(p2_metric).max().alias("peak_" + p2_metric),
            )

    def get_lateral_pos_at_cc_hit_time(self, condition=None, collapsed=True):
        if collapsed:
            df = self.collapsed_trial_df
        else:
            df = self.trial_df

        if condition is not None:
            df = df.filter(pl.col("condition") == condition)
        
        dff = (
            df.filter(pl.col("perturbation_type").str.contains("perturbation"))
            # Flip left perturbations to be positive
            .with_columns(
                lateral_pos_at_cc_hit_time=pl.when(pl.col("perturbation_type") == "perturbation_left")
                .then(pl.col("lateral_pos_at_cc_hit_time").mul(-1))
                .otherwise(pl.col("lateral_pos_at_cc_hit_time"))
            )
            # Take mean across left and right perturbations
            .group_by(pl.col("subject", "condition"), maintain_order=True)
            .agg(pl.col("lateral_pos_at_cc_hit_time").drop_nans().mean())
            # Flip left subjects back to be positive
            .with_columns(
                lateral_pos_at_cc_hit_time=pl.when(pl.col("subject").str.contains("L"))
                .then(pl.col("lateral_pos_at_cc_hit_time").mul(-1))
                .otherwise(pl.col("lateral_pos_at_cc_hit_time"))
            )
        )
        return dff

    def _handle_opposite_jump(self, df, col_name):
        df = df.with_columns(
            pl.when(pl.col("condition").str.contains("opposite_jump"))
            .then(pl.col(col_name).mul(-1))
            .otherwise(pl.col(col_name))
        )
        return df


class Models:
    def __init__(
        self,
        regular_df: pl.DataFrame,
        probe_df: pl.DataFrame,
        conditions,
        alphas: list[float] = [0.0, 0.0, 1.0, 0.5],
        partner_knowledges: list[bool] = [False, True, True, True],
    ):
        self.regular_df = regular_df
        self.probe_df = probe_df.with_columns(pl.col(pl.Float64()).round(3))
        self.conditions = conditions
        # self.models = models

        self.alphas = alphas
        self.partner_knowledges = partner_knowledges
        max_trial_number = probe_df["experiment_trial_number"].max()
        self.max_exp_trial_number = max_trial_number if max_trial_number is not None else 0
        self.cropped_probe_df = self._crop_probe_df()

    def _crop_probe_df(self):
        before_probe_time = 50  # 50ms
        after_probe_time = 400  # 400ms
        extra_time = 0  # 100ms so that none of the models get cut off and can't vstack below
        time_from_probe_onset = np.arange(-before_probe_time, after_probe_time + 10, 10).round(2)
        # * Crop probe df
        df_list = []
        for alpha, partner_knowledge in zip(self.alphas, self.partner_knowledges):
            for i, condition in enumerate(self.conditions):
                for j in range(1, self.max_exp_trial_number + 1):
                    dff = self.probe_df.filter(
                        pl.col("alpha") == alpha,
                        pl.col("partner_knowledge") == partner_knowledge,
                        pl.col("condition") == condition,
                        pl.col("experiment_trial_number") == j,
                    )
                    start_index = np.round(
                        dff["jump_time"][0] - before_probe_time, 2
                    )  # Need to round so that the is_between works correctly
                    end_index = np.round(dff["jump_time"][0] + after_probe_time + extra_time, 2)
                    filt_df = dff.filter(pl.col("timepoint").is_between(start_index, end_index, closed="both"))
                    assert len(filt_df) == 46  # 50ms before and 400ms after the probe onset
                    # * Add time from probe onset column
                    filt_df = filt_df.with_columns(
                        time_from_probe_onset=time_from_probe_onset,
                        p1_applied_force_double=pl.col("p1_applied_force").mul(2),
                    )
                    # if len(filt_df)<47:
                    #     print(alpha, partner_knowledge, i)
                    df_list.append(filt_df)
        return pl.concat(df_list)

    def lateral_deviation(self):
        pass

    def involuntary_feedback_response(self):
        pass

    def onset_times(self):
        pass
