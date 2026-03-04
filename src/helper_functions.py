import numpy as np  
import polars as pl
import analysis_utilities as au

def groupby_trial_to_numpy(df, metric):
    return np.vstack(df.group_by("Experiment Trial Number", maintain_order=True).agg(pl.col(metric))[metric].to_numpy())

def get_left_minus_right_force(df, condition, exp) -> tuple[np.ndarray, np.ndarray]:
    df_rightward_probe = df.filter((pl.col("perturbation_type").str.contains("probe_right")) & pl.col("target_type").str.contains(condition))
    df_leftward_probe = df.filter((pl.col("perturbation_type").str.contains("probe_left")) & pl.col("target_type").str.contains(condition))
    if exp == 'Exp1':
        rhfx = (df_leftward_probe['filtered_right_fs_forcex'] - df_rightward_probe['filtered_right_fs_forcex']).to_numpy() 
        lhfx = (df_rightward_probe['filtered_left_fs_forcex'] - df_leftward_probe['filtered_left_fs_forcex']).to_numpy() #! Flipped bc right probe is actually left for p2
    elif exp == 'Exp2':
        rhfx = (df_rightward_probe['filtered_right_fs_forcex'] - df_leftward_probe['filtered_right_fs_forcex']).to_numpy() 
        lhfx = (df_leftward_probe['filtered_left_fs_forcex'] - df_rightward_probe['filtered_left_fs_forcex']).to_numpy() #! Flipped bc right probe is actually left for p2
    return rhfx, lhfx

def collapsed_left_minus_right_force(df, exp):
    '''
    This takes the sub_avg_df ONLY. Which means we average across all P1s and all P2s for each condition and trial type.
    
    Need to then average across when p1 is relevant and p2 is relevant
    - Then just joint relevant and joint irrelevant
   
    So will return four conditions
    1. Both Irrelevant
    2. Self Irrelevant, Partner Relevant
    3. Self Relevant, Partner Irrelevant
    4. Both Relevant 
    
    Remember that "probe_right" is relative to p1 so thats why leftward_probe_p2 is the probe_right filter
    '''
    if exp == "Exp1":
        C = 1
    elif exp == "Exp2":
        C = -1
    #* Self relevant
    df_leftward_probe_p1 = df.filter((pl.col("perturbation_type").str.contains("probe_left")) & pl.col("target_type").str.contains("p1_relevant"))
    df_rightward_probe_p1 = df.filter((pl.col("perturbation_type").str.contains("probe_right")) & pl.col("target_type").str.contains("p1_relevant"))
    p1_fx = C*(df_leftward_probe_p1['filtered_right_fs_forcex'] - df_rightward_probe_p1['filtered_right_fs_forcex']).to_numpy() 
    
    df_leftward_probe_p2 = df.filter((pl.col("perturbation_type").str.contains("probe_right")) & pl.col("target_type").str.contains("p2_relevant"))
    df_rightward_probe_p2 = df.filter((pl.col("perturbation_type").str.contains("probe_left")) & pl.col("target_type").str.contains("p2_relevant")) # "probe_left" here because this is rightward probe for p2
    p2_fx = C*(df_leftward_probe_p2['filtered_left_fs_forcex'] - df_rightward_probe_p2['filtered_left_fs_forcex']).to_numpy() 
    
    self_relevant_fx = (p1_fx + p2_fx) / 2 
    
    #* Partner relevant
    df_leftward_probe_p1 = df.filter((pl.col("perturbation_type").str.contains("probe_left")) & pl.col("target_type").str.contains("p2_relevant"))
    df_rightward_probe_p1 = df.filter((pl.col("perturbation_type").str.contains("probe_right")) & pl.col("target_type").str.contains("p2_relevant"))
    p1_fx = C*(df_leftward_probe_p1['filtered_right_fs_forcex'] - df_rightward_probe_p1['filtered_right_fs_forcex']).to_numpy() 
    
    df_leftward_probe_p2 = df.filter((pl.col("perturbation_type").str.contains("probe_right")) & pl.col("target_type").str.contains("p1_relevant"))
    df_rightward_probe_p2 = df.filter((pl.col("perturbation_type").str.contains("probe_left")) & pl.col("target_type").str.contains("p1_relevant")) # "probe_left" here because this is rightward probe for p2
    p2_fx = C*(df_leftward_probe_p2['filtered_left_fs_forcex'] - df_rightward_probe_p2['filtered_left_fs_forcex']).to_numpy() 
    
    partner_relevant_fx = (p1_fx + p2_fx) / 2 
    
    #* Joint Irrelevant
    df_leftward_probe = df.filter((pl.col("perturbation_type").str.contains("probe_left")) & pl.col("target_type").str.contains("joint_irrelevant"))
    df_rightward_probe = df.filter((pl.col("perturbation_type").str.contains("probe_right")) & pl.col("target_type").str.contains("joint_irrelevant"))
    p1_fx = C*(df_leftward_probe['filtered_right_fs_forcex'] - df_rightward_probe['filtered_right_fs_forcex']).to_numpy() 
    p2_fx = C*(df_rightward_probe['filtered_left_fs_forcex'] - df_leftward_probe['filtered_left_fs_forcex']).to_numpy() #! Flipped bc right probe is actually left for p2
    joint_irrelevant_fx = (p1_fx + p2_fx) /2
    
    #* Joint Relevant
    df_leftward_probe = df.filter((pl.col("perturbation_type").str.contains("probe_left")) & pl.col("target_type").str.contains("joint_relevant"))
    df_rightward_probe = df.filter((pl.col("perturbation_type").str.contains("probe_right")) & pl.col("target_type").str.contains("joint_relevant"))
    p1_fx = C*(df_leftward_probe['filtered_right_fs_forcex'] - df_rightward_probe['filtered_right_fs_forcex']).to_numpy() 
    p2_fx = C*(df_rightward_probe['filtered_left_fs_forcex'] - df_leftward_probe['filtered_left_fs_forcex']).to_numpy() #! Flipped bc right probe is actually left for p2
    joint_relevant_fx = (p1_fx + p2_fx) /2
    
    return (joint_irrelevant_fx, partner_relevant_fx, self_relevant_fx, joint_relevant_fx)

def run_mean_comparisons(df, metric_name, collapsed_condition_names, combos=["01","12","23","13"], 
                         test="mean", alternative="two-sided", ):
    pvals = dict(zip(combos,[2,2,2,2]))
    cles = dict(zip(combos,[2,2,2,2]))
    for i,combo in enumerate(combos):
        a = int(combo[0])
        b = int(combo[1])
        metric1 = df.filter(pl.col("condition") == collapsed_condition_names[a])[metric_name].to_numpy()
        metric2 = df.filter(pl.col("condition") == collapsed_condition_names[b])[metric_name].to_numpy()
        pvals[combo],dist = au.bootstrap(metric1, metric2,
                            return_distribution=True,
                            paired=True, M=1e6, test=test, alternative=alternative)
        
        cles[combo] = au.cles(metric1, metric2)
        if cles[combo]<50.0:
            cles[combo] = 100 - cles[combo]
            
    pvals_corrected = dict(zip(pvals.keys(), au.holmbonferroni_correction(list(pvals.values()))))    
    return pvals, pvals_corrected, cles

def generate_Q(weight_dict, state_mapping, cross_terms:list[tuple], QVAL):
    # If k is in cross_terms, then leave diagonal as 1 and do the cross_terms together
    assert len(weight_dict) == len(state_mapping)
    assert list(weight_dict.keys()) == list(state_mapping.keys())
    
    Q = np.zeros((len(state_mapping),len(state_mapping)))
    for k,v in weight_dict.items():
        # Check for Off-Diagonals
        pair_check = [pair for pair in cross_terms if k in pair] 
        if len(pair_check)>0:
            for pair in pair_check:
                k1,k2 = pair
                # Q[state_mapping[k1], state_mapping[k1]] = v*QVAL
                # Q[state_mapping[k2], state_mapping[k2]] = v*QVAL
                
                Q[state_mapping[k1], state_mapping[k2]] = -v*QVAL
                Q[state_mapping[k2], state_mapping[k1]] = -v*QVAL
        # else:
        Q[state_mapping[k], state_mapping[k]] = v*QVAL
    return Q

def create_observation_matrix(state_mapping, observable_states):
    nx = len(state_mapping) # Num states
    nz = len(observable_states) # num sensory states
    C = np.zeros((nz,nx))
    row = 0
    for key,idx in state_mapping.items():
        if idx in observable_states:
            C[row, idx] = 1
            row+=1
    return C

