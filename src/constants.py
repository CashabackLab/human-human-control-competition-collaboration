from pathlib import Path
import polars as pl
import data_visualization as dv
import matplotlib as mpl

wheel = dv.ColorWheel()

stylelib_path = r"C:\Users\Seth Sullivan\miniconda3\envs\aim2\Lib\site-packages\matplotlib\mpl-data\stylelib"
laptop_data_path = Path(r"C:\Users\sully\OneDrive - University of Delaware - o365\Subject_Data\Aim3_Exp1")
if laptop_data_path.exists():
    RAW_DATA_PATH = laptop_data_path
    DATA_PATH = Path(r"C:\Users\sully\OneDrive\Desktop\PhD Research\Aim2")
else:
    RAW_DATA_PATH = Path(r"D:\OneDrive - University of Delaware - o365\Subject_Data\Aim3_Exp1")
    DATA_PATH = Path(r"D:\OneDrive - University of Delaware - o365\Desktop\Aim3\data")

with open(RAW_DATA_PATH / "fields_pull.txt") as f:
    column_names = [line.rstrip() for line in f]

TOTAL_TRIALS = 774

TRIAL_TYPE_EVENT_CODES = [
    "E_PERTURBATION_RIGHT",
    "E_PERTURBATION_LEFT",
    "E_PROBE_RIGHT",
    "E_PROBE_LEFT",
    "E_PROBE_NEUTRAL",
    "E_REGULAR_TRIAL",
]

TRIAL_START_EVENT_CODES = [
    "E_SHOW_START",
    "E_BOTH_HANDS_IN_START",
    "E_TRIAL_START_SIGNAL",
    "E_TRIAL_COMPLETED",
]

kinematic_columns = [
    "Right_HandX",
    "Right_HandY",
    "Right_HandXVel",
    "Right_HandYVel",
    "Right_FS_ForceX",
    "Right_FS_ForceY",
    "Left_HandX",
    "Left_HandY",
    "Left_HandXVel",
    "Left_HandYVel",
    "Left_FS_ForceX",
    "Left_FS_ForceY",
    "p1_center_cursor_pos_x",
    "p1_center_cursor_pos_y",
    "p2_center_cursor_pos_x",
    "p2_center_cursor_pos_y",
]

# Not filtering the center cursor or targets
filtered_kinematic_columns = [
    "filtered_right_handx",
    "filtered_right_handy",
    "filtered_right_handxvel",
    "filtered_right_handyvel",
    "filtered_right_fs_forcex",
    "filtered_left_handx",
    "filtered_left_handy",
    "filtered_left_handxvel",
    "filtered_left_handyvel",
    "filtered_left_fs_forcex",
    "p1_center_cursor_pos_x",
    "p1_center_cursor_pos_y",
    "p2_center_cursor_pos_x",
    "p2_center_cursor_pos_y",
    "p1_self_target_pos_x",
    "p1_partner_target_pos_x",
    "p2_self_target_pos_x",
    "p2_partner_target_pos_x",
    #  "p1_cursor_pos_x","p1_cursor_pos_y",
    #  "p2_cursor_pos_x","p2_cursor_pos_y",
    #  "center_cursor_pos_x","center_cursor_pos_y"
]

perturbation_type_map = {
    0: "regular",
    1: "perturbation_right",
    -1: "perturbation_left",
    2: "probe_right",
    -2: "probe_left",
    3: "probe_neutral",
}
perturbation_types = [v for k, v in perturbation_type_map.items()]

# lighten_nums = [0.1, 0.5, 1.3, 0.9]
# cm = mpl.colormaps["Wistia"]
# vecx = [0.25,0.9, 0.5]
# condition_colors_dark = [cm(x/len(vecx) + 0.4) for x in range(len(vecx))]

josh_brown = "#834300"
light_gray = "#CECECE"
josh_oranges = ["#FFC482", "#FD8B0B", "#C57321", josh_brown]
blues = ["#96BEDC", "#6482A0", "#3C5A78", "#1E324B"]
purples = ["#E7C9FF", "#D5A2EF", "#8B5BC8", "#A04E9E"]
oranges = ["#FFB84D", "#FF9E3D", "#FF7F2A", "#C65A1E"]
green_grey_navy = ["#f0ead2", "#87bba2", "#5bc0be", "#237dc6"]

yellow_red = ["#ffd97d", "#ffba08", "#ee6352", "#d10000"]
blues2 = ["#fedfd4", "#9dcee2", "#4091c9", "#1368aa"]
greenyellow_blue = ["#d8f3dc", "#d9ed92", "#76c893", "#34a0a4"]
greenyellow = ["#b7e4c7", "#d9ed92", "#76c893", "#007f5f"]

liv_pink = ["#F6C8DE", "#EC87B9", "#DC267F", "#98295D"]
liv_orange = ["#FE9859", "#dd4b1a", "#FE6100"]  # , "#B9521A"]
liv_greens = ["#b0f2ba", "#66d17f", "#2aa84a", "#256e34"]
liv_purples = ["#c18bcc", "#923aa1", "#64256e"]

darken_nums = [3, 1.3, 1.1, 1.1]
lighten_nums = [0.6, 1.0, 1.3]
condition_colors_dark = [
    wheel.lighten_color(color, lighten_num) for lighten_num, color in zip(lighten_nums, liv_purples)
] + [wheel.lighten_color(color, lighten_num) for lighten_num, color in zip(lighten_nums, liv_greens)]
condition_colors_light = [
    wheel.lighten_color(color, lighten_num) for lighten_num, color in zip(lighten_nums, liv_purples)
] + [wheel.lighten_color(color, lighten_num) for lighten_num, color in zip([1.2, 1.2, 1.2], liv_greens)]
condition_colors_light = [wheel.lighten_color(wheel.vibrant_red, lighten_num) for lighten_num in lighten_nums] + [
    wheel.lighten_color(wheel.rak_blue, lighten_num) for lighten_num in lighten_nums
]
condition_colors_dark_alt = [
    condition_colors_dark[3],
    condition_colors_dark[0],
    condition_colors_dark[4],
    condition_colors_dark[1],
    condition_colors_dark[5],
    condition_colors_dark[2],
]
condition_colors_light_alt = [
    condition_colors_light[3],
    condition_colors_light[0],
    condition_colors_light[4],
    condition_colors_light[1],
    condition_colors_light[5],
    condition_colors_light[2],
]

red = "#D21F3C"
raspberry_red = "#D21F3C"
blue = "#2D5DA1"
medium_sapphire = "#2D5DA1"
nice_green = "#269999"
self_color = wheel.lighten_color(wheel.grey, 1.4)
self_color_light = wheel.dark_grey
partner_color = wheel.lighten_color(wheel.grey, 0.7)  #
cc_color = wheel.rak_orange

# * Target Information
# TODO Change this when I upload new data
if True:
    row_names = pl.Series(
        "label",
        [
            "p1_cursor",
            "p2_cursor",
            "center_cursor",
            "p1_target",
            "p2_target",
            "end_target",
            "none",
            "p1_start",
            "p2_start",
            "feedback_target",
        ],
    )
    # TODO come up with a better fix for this
    trial_table_df = pl.read_csv(
        RAW_DATA_PATH / "task_tables" / "Sub1Trial_Table.csv"
    )  # All task tables are the same so we're good
    Jump_Cross_Y_Dist_From_Start = trial_table_df.select(pl.col("Jump_Cross_Y_Dist_From_Start"))[0].item() / 100

    target_table_df = pl.read_csv(
        RAW_DATA_PATH / "task_tables" / "Sub1Target_Table.csv"
    )  # All task tables are the same so we're good
    target_table_df.insert_column(0, row_names)

    # Target Dimensions
    start_radius = target_table_df.filter(pl.col("label") == "p1_start")["Dimension 1"].item() / 100
    target_width = target_table_df.filter(pl.col("label") == "p1_target")["Dimension 1"].item() / 100
    target_height = target_table_df.filter(pl.col("label") == "p1_target")["Dimension 2"].item() / 100

    # Target Positions
    p1_startx = target_table_df.filter(pl.col("label") == "p1_start")["X"].item() / 100
    p1_starty = target_table_df.filter(pl.col("label") == "p1_start")["Y"].item() / 100
    p2_startx = target_table_df.filter(pl.col("label") == "p2_start")["X"].item() / 100
    p2_starty = target_table_df.filter(pl.col("label") == "p2_start")["Y"].item() / 100
    distance_between_hands = p1_startx - (-p2_startx)

    target_x = target_table_df.filter(pl.col("label") == "p1_target")["X"].item() / 100
    target_y = target_table_df.filter(pl.col("label") == "p1_target")["Y"].item() / 100

    target_x_corner = target_x - 0.5 * target_width
    target_y_corner = target_y - 0.5 * target_height

condition_labels = [
    "Collaborative\nTarget Jump",
    "Competitive Self\nTarget Jump",
    "Competitive Opponent\nTarget Jump",
    "Solo Both\nTarget Jump",
    "Solo Self\nTarget Jump",
    "Solo Other\nTarget Jump",
]
condition_labels_alt = [
    condition_labels[3],
    condition_labels[0],
    condition_labels[4],
    condition_labels[1],
    condition_labels[5],
    condition_labels[2],
]
condition_names = [
    "joint_same_jump",
    "joint_p1_jump",
    "joint_p2_jump",
    "solo_same_jump",
    "solo_p1_jump",
    "solo_p2_jump",
]
condition_names_alt = [
    condition_names[3],
    condition_names[0],
    condition_names[4],
    condition_names[1],
    condition_names[5],
    condition_names[2],
]
condition_names_collapsed = [
    "joint_same_jump",
    "joint_self_jump",
    "joint_partner_jump",
    "solo_same_jump",
    "solo_self_jump",
    "solo_partner_jump",
]
condition_names_collapsed_alt = [
    condition_names_collapsed[3],
    condition_names_collapsed[0],
    condition_names_collapsed[4],
    condition_names_collapsed[1],
    condition_names_collapsed[5],
    condition_names_collapsed[2],
]
p1_p2_jump_type = [
    {"p1": "joint_same_jump", "p2": "joint_same_jump"},  # Joint Same jump
    {"p1": "joint_p1_jump", "p2": "joint_p2_jump"},  # Joint Self jump
    {"p1": "joint_p2_jump", "p2": "joint_p1_jump"},  # Joint Partner jump
    {"p1": "solo_same_jump", "p2": "solo_same_jump"},  # Solo Same jump
    {"p1": "solo_p1_jump", "p2": "solo_p2_jump"},  # Solo Self jump
    {"p1": "solo_p2_jump", "p2": "solo_p1_jump"},  # Solo Partner jump
]
collapsed_conditions_to_p1_p2 = dict(zip(condition_names_collapsed, p1_p2_jump_type))
# = [
#     "self_jump",
#     "same_jump",
#     "partner_jump",
#     "opposite_jump",
# ]

model_names = ["competitive", "partial_competitive", "partial_cooperative", "cooperative"]

BLOCK_ORDER = {
    "Sub1": {
        2: "solo_same_jump",
        3: "solo_same_jump",
        5: "solo_opposite_jump",
        6: "solo_opposite_jump",
        8: "joint_same_jump",
        9: "joint_same_jump",
        11: "joint_opposite_jump",
        12: "joint_opposite_jump",
    }
}
