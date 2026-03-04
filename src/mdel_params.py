import numpy as np
import src.helper_functions as hf

h=0.01
sensory_delay = 130 # ms converted to timesteps in program
sensory_delay_steps = int(sensory_delay*(0.001/h))
#* Timesteps
# Since the model doesn't go until sensory delay
VIA_T = 0.52 + sensory_delay/1000 # add sensory delay bc I counted participants time not including sensory delay

FINAL_T = VIA_T + 0.2 # 200ms to stop
timesteps = np.arange(0,FINAL_T,h)
VIA_N = np.argwhere(np.abs(timesteps - VIA_T) < 0.00001)[0,0]
N = len(timesteps)

b,m = 0.1,1
tau = int(20*(0.001/h)) # 20 ms  
probe_duration = 250 # ms, converted to timesteps in program
probe_duration_steps = int(probe_duration*(0.001/h)) 
perturbation_distance = -0.02 # 2cm jumps
hold_time = 0
p1_x0, p1_y0 = 0.10, 0
p2_x0, p2_y0 = -0.10, 0

x0 = np.array([
    [p1_x0], # right px
    [0], # right vx
    [0], # right Fx
    [p1_y0], # right py
    [0], # right vy
    [0], # right Fy
    [p2_x0], # left px
    [0], # left vx
    [0], # left Fx
    [p2_y0], # left py
    [0], # left vy
    [0], # left Fy
    [((p1_x0+p2_x0)/2)], # center cursor px
    [(p1_y0+p2_y0)/2], # center cursor py
    [((p1_x0+p2_x0)/2)], # right target px
    [0.2], # right target py
    [((p1_x0+p2_x0)/2)], # left target px
    [0.2], # left target py
    [0.25], # final target y position, no x bc i don't care where the hands are in the x-dimension
])
state_mapping = {
    "rhx":0,
    "rhvx":1,
    "rfx":2,
    "rhy":3,
    "rhvy":4,
    "rfy":5,
    "lhx":6,
    "lhvx":7,
    "lfx":8,
    "lhy":9,
    "lhvy":10,
    "lfy":11,
    "ccx":12,
    "ccy":13,
    "rtx":14,
    "rty":15,
    "ltx":16,
    "lty":17,
    'fty':18,
}
x = -b/m
y = 1/m
z = -1/tau
A_joint = np.block([
    #rhx #rhvx  #rFx  #rhy  #rhvy #rFy #lhx  #lhvx #lFx #lhy #lhvy #lhFy 
    [0,    1,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhx-position change
    [0,    x,    y,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhx-velocity change
    [0,    0,    z,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rFx change
    [0,    0,    0,   0,    1,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhy-position change
    [0,    0,    0,   0,    x,    y,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhy-velocity change
    [0,    0,    0,   0,    0,    z,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rFy change
    [0,    0,    0,   0,    0,    0,     0,   1,    0,    0,     0,    0,   np.zeros(7)],  # How does lhx-position change
    [0,    0,    0,   0,    0,    0,     0,   x,    y,    0,     0,    0,   np.zeros(7)],  # How does lhx-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    z,    0,     0,    0,   np.zeros(7)],  # How does lFx change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     1,    0,   np.zeros(7)],  # How does lhy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     x,    y,   np.zeros(7)],  # How does lhy-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    z,   np.zeros(7)],  # How does lFy change
    [0,  0.5,    0,   0,    0,    0,     0, 0.5,    0,    0,     0,    0,   np.zeros(7)],  # How does cx-position change (changes based on the VELOCITY of rhvx and lhvx)
    [0,    0,    0,   0,  0.5,    0,     0,   0,    0,    0,     0.5,  0,   np.zeros(7)],  # How does cy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does righthand target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does righthand target y change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lefthand target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does leftand target y change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does final bar target y change
])
A_solo = np.block([
    #rhx #rhvx  #rFx  #rhy  #rhvy #rFy #lhx  #lhvx #lFx #lhy #lhvy #lhFy 
    [0,    1,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhx-position change
    [0,    x,    y,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhx-velocity change
    [0,    0,    z,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rFx change
    [0,    0,    0,   0,    1,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhy-position change
    [0,    0,    0,   0,    x,    y,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhy-velocity change
    [0,    0,    0,   0,    0,    z,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rFy change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lhx-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lhx-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lFx change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     1,    0,   np.zeros(7)],  # How does lhy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     x,    y,   np.zeros(7)],  # How does lhy-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    z,   np.zeros(7)],  # How does lFy change
    [0,  0.5,    0,   0,    0,    0,     0, 0.5,    0,    0,     0,    0,   np.zeros(7)],  # How does cx-position change (changes based on the VELOCITY of rhvx and lhvx)
    [0,    0,    0,   0,    1,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does cy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does righthand target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does righthand target y change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lefthand target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does leftand target y change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does final bar target y change
])
A_probe_joint = np.block([
    #rhx #rhvx  #rFx  #rhy  #rhvy #rFy #lhx  #lhvx #lFx #lhy #lhvy #lhFy 
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhx-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhx-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rFx change
    [0,    0,    0,   0,    1,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhy-position change
    [0,    0,    0,   0,    x,    y,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhy-velocity change
    [0,    0,    0,   0,    0,    z,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rFy change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lhx-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lhx-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lFx change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     1,    0,   np.zeros(7)],  # How does lhy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     x,    y,   np.zeros(7)],  # How does lhy-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    z,   np.zeros(7)],  # How does lFy change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does cx-position change (changes based on the VELOCITY of rhvx and lhvx)
    [0,    0,    0,   0,  0.5,    0,     0,   0,    0,    0,     0.5,  0,   np.zeros(7)],  # How does cy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does righthand target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does righthand target y change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lefthand target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does leftand target y change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does final bar target y change
])
A_probe_solo = np.block([
    #rhx #rhvx  #rFx  #rhy  #rhvy #rFy #lhx  #lhvx #lFx #lhy #lhvy #lhFy 
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhx-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhx-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rFx change
    [0,    0,    0,   0,    1,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhy-position change
    [0,    0,    0,   0,    x,    y,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rhy-velocity change
    [0,    0,    0,   0,    0,    z,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does rFy change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lhx-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lhx-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lFx change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     1,    0,   np.zeros(7)],  # How does lhy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     x,    y,   np.zeros(7)],  # How does lhy-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    z,   np.zeros(7)],  # How does lFy change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does cx-position change (changes based on the VELOCITY of rhvx and lhvx)
    [0,    0,    0,   0,    1,    0,     0,   0,    0,    0,     0,  0,   np.zeros(7)],  # How does cy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does righthand target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does righthand target y change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does lefthand target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does leftand target y change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(7)],  # How does final bar target y change
])

B1 = np.array([
    #Frx  Fry  
    [ 0,    0],  # applying to rhx-pos
    [ 0,    0],  # applying to rhx-vel
    [ 1/tau,    0],  # applying to rFx
    [ 0,    0],  # applying to rhy-pos
    [ 0,    0],  # applying to rhy-vel
    [ 0,    1/tau],  # applying to rFy
    
    [ 0,    0],  # applying to lhx-pos
    [ 0,    0],  # applying to lhx-vel
    [ 0,    0],  # applying to lFx
    [ 0,    0],  # applying to lhy-pos
    [ 0,    0],  # applying to lhy-vel
    [ 0,    0],  # applying to lFy

    [ 0,    0],  # applying to cx-pos
    [ 0,    0],  # applying to cy-pos
    
    [ 0,    0],  # applying to righthand target x
    [ 0,    0],  # applying to righthand target y
    [ 0,    0],  # applying to lefthand target x
    [ 0,    0],  # applying to lefthand target y
    [ 0,    0],  # applying to finalbar target y
])

B2 = np.array([
    #Flx    Fly  
    [ 0,    0],  # applying to rhx-pos
    [ 0,    0],  # applying to rhx-vel
    [ 0,    0],  # applying to rFx
    [ 0,    0],  # applying to rhy-pos
    [ 0,    0],  # applying to rhy-vel
    [ 0,    0],  # applying to rFy
    
    [ 0,    0],  # applying to lhx-pos
    [ 0,    0],  # applying to lhx-vel
    [ 1/tau,    0],  # applying to lFx
    [ 0,    0],  # applying to lhy-pos
    [ 0,    0],  # applying to lhy-vel
    [ 0,    1/tau],  # applying to lFy

    [ 0,    0],  # applying to cx-pos
    [ 0,    0],  # applying to cy-pos
    
    [ 0,    0],  # applying to righthand target x
    [ 0,    0],  # applying to righthand target y
    [ 0,    0],  # applying to lefthand target x
    [ 0,    0],  # applying to lefthand target y
    [ 0,    0],  # applying to finalbar target y
])

p1_observable_states = [
    state_mapping["rhx"], state_mapping["rhy"],
    state_mapping["rhvx"], state_mapping["rhvy"],
    state_mapping["rfx"], state_mapping["rfy"],
    state_mapping["lhx"], state_mapping["lhy"],
    state_mapping["lhvx"], state_mapping["lhvy"],
    state_mapping["ccx"], state_mapping["ccy"],
    state_mapping["rtx"], state_mapping["rty"],
    state_mapping["ltx"], state_mapping["lty"],
    state_mapping['fty']
]
p2_observable_states = [
    state_mapping["rhx"], state_mapping["rhy"],
    state_mapping["rhvx"], state_mapping["rhvy"],
    state_mapping["lfx"], state_mapping["lfy"],
    state_mapping["lhx"], state_mapping["lhy"],
    state_mapping["lhvx"], state_mapping["lhvy"],
    state_mapping["ccx"], state_mapping["ccy"],
    state_mapping["rtx"], state_mapping["rty"],
    state_mapping["ltx"], state_mapping["lty"],
    state_mapping['fty']
]
# assert len(p1_observable_states) == len(p2_observable_states)
C1 = np.zeros((len(p1_observable_states),A_joint.shape[0]))
C2 = np.zeros((len(p2_observable_states),A_joint.shape[0]))
row1 = 0
row2 = 0
for key,idx in state_mapping.items():
    if idx in p1_observable_states:
        C1[row1, idx] = 1
        row1+=1
    
    if idx in p2_observable_states:
        C2[row2, idx] = 1
        row2+=1
        
Q = np.array(
    [
        #rhx  rhvx rfx  rhy rhvy  rFy  lhx  lhvx  lFx  lhy  lhvy lFy  ccx ccy rtx  rty ltx lty fty
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0], # rhx 0
        [0,   1,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0], # rhvx 1
        [0,   0,   1,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0], # rFx 2
        [0,   0,   0,    0,   0,   0,    0,   0,   0,  -0,   0,   0,   0,  0,  0,  0,   0,  0,  0], # rhy 3
        [0,   0,   0,    0,   1,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0], # rhvy 4
        [0,   0,   0,    0,   0,   1,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0], # rFy 5
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0], # lhx 6
        [0,   0,   0,    0,   0,   0,    0,   1,   0,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0],  # lhvx 7
        [0,   0,   0,    0,   0,   0,    0,   0,   1,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0],  # lFx 8
        [0,   0,   0,   -0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0,   0,  0,  0],  # lhy 9
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   1,   0,   0,  0,  0,  0,   0,  0,  0],  # lhvy 10
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   1,   0,  0,  0,  0,   0,  0,  0],  # lFy 11
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   1,  0, -1,  0,   -1, 0,  0], # ccx 12
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  1,  0, -1,   0, -1, -1],  # ccy 13
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,  0,    0,  -1,  0,  1,  0,   0,  0,  0],  # rtx 14
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,  0,   0,   0,  -1,  0,  1,   0,  0,  0],  # rty 15
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,  0,    0,  -1,  0,  0,  0,   1,  0,  0],  # ltx 16
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,  0,   0,   0,  -1,  0,  0,   0,  1,  0],  # lty 17
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,  0,   0,   0,  -1,  0,  0,   0,  0,  1],  # fty 18
    ]
)

w_viapoint = 1.5

Q1_VIA_WEIGHTS = {
    'rhx': 0,
    'rhvx': 0,
    'rfx': 0,
    'rhy': 0,
    'rhvy': 0,
    'rfy': 0,
    'lhx': 0,
    'lhvx': 0,
    'lfx': 0,
    'lhy': 0,
    'lhvy': 0,
    'lfy': 0,
    'ccx': w_viapoint,
    'ccy': w_viapoint,
    'rtx': w_viapoint,
    'rty': w_viapoint,
    'ltx': 0,
    'lty': 0,
    'fty': 0,
}
Q2_VIA_WEIGHTS = {
    'rhx': 0,
    'rhvx': 0,
    'rfx': 0,
    'rhy': 0,
    'rhvy': 0,
    'rfy': 0,
    'lhx': 0,
    'lhvx': 0,
    'lfx': 0,
    'lhy': 0,
    'lhvy': 0,
    'lfy': 0,
    'ccx': w_viapoint,
    'ccy': w_viapoint,
    'rtx': 0,
    'rty': 0,
    'ltx': w_viapoint,
    'lty': w_viapoint,
    'fty': 0,
}
QVAL = 1e4
Q1_VIA = hf.generate_Q(
    weight_dict=Q1_VIA_WEIGHTS, 
    state_mapping=state_mapping, 
    cross_terms=[["ccx","rtx"], ["ccy","rty"]], 
    QVAL=QVAL
)
Q2_VIA = hf.generate_Q(
    weight_dict=Q2_VIA_WEIGHTS, 
    state_mapping=state_mapping, 
    cross_terms=[["ccx","ltx"], ["ccy","lty"]], 
    QVAL = QVAL,
)

# NOTE Don't need final target per say. Just need to punish velocity properly at final timestep
w_targety = 0.01
w_vel = 0.0001
w_force = 0.00
Q1N_WEIGHTS = {
    'rhx': 0,
    'rhvx': w_vel,
    'rfx': 0,
    'rhy': w_targety,
    'rhvy': w_vel,
    'rfy': 0,
    'lhx': 0,
    'lhvx': 0,
    'lfx': 0,
    'lhy': 0,
    'lhvy': 0,
    'lfy': 0,
    'ccx': 0,
    'ccy': 0,
    'rtx': 0,
    'rty': 0,
    'ltx': 0,
    'lty': 0,
    'fty': w_targety,
}
Q2N_WEIGHTS = {
    'rhx': 0,
    'rhvx': 0,
    'rfx': 0,
    'rhy': 0,
    'rhvy': 0,
    'rfy': 0,
    'lhx': 0,
    'lhvx': w_vel,
    'lfx': 0,
    'lhy': w_targety,
    'lhvy': w_vel,
    'lfy': 0,
    'ccx': 0,
    'ccy': 0,
    'rtx': 0,
    'rty': 0,
    'ltx': 0,
    'lty': 0,
    'fty': w_targety,
}
Q1N = hf.generate_Q(
    weight_dict=Q1N_WEIGHTS, 
    state_mapping=state_mapping, 
    cross_terms=[["rhy","fty"]], 
    QVAL=QVAL
)

Q2N = hf.generate_Q(
    weight_dict=Q2N_WEIGHTS, 
    state_mapping=state_mapping, 
    cross_terms=[["lhy","fty"]], 
    QVAL = QVAL,
)

# Set up time varying Q
Q1 = np.zeros((N+1, *A_joint.shape))
Q1[VIA_N] = Q1_VIA
Q1[N] = Q1N
Q2 = np.zeros((N+1, *A_joint.shape))
Q2[VIA_N] = Q2_VIA
Q2[N] = Q2N

RVAL = 1e-4
R11 = np.eye(B1.shape[1])*RVAL
R22 = np.eye(B2.shape[1])*RVAL
R12 = np.eye(B1.shape[1])*RVAL
R21 = np.eye(B2.shape[1])*RVAL

#* Create noise and covariance matrices

INTERNAL_MODEL_NOISE = 1e-5 # Should only affect current prediction, not previous augmented ones
PROCESS_NOISE = 1e-2
sig_pos = 0.0001
sig_vel = 0.001
sig_f = 0.1
MEASUREMENT_NOISE_MOD1 =  np.array([[
    sig_pos, # right px
    sig_vel, # right vx
    sig_f, # right fx
    sig_pos, # right py
    sig_vel, # right vy
    sig_f, # right fy
    sig_pos, # left px
    sig_vel, # left vx
    sig_pos, # left py
    sig_vel, # left vy
    sig_pos, # center cursor px
    sig_pos, # center cursor py
    sig_pos, # rh target px
    sig_pos, # rh target py
    sig_pos, # lh target px
    sig_pos, # lh target py
    sig_pos, # finaltarget py
]]).T
MEASUREMENT_NOISE_MOD2 =  np.array([[
    sig_pos, # right px
    sig_vel, # right vx
    sig_pos, # right py
    sig_vel, # right vy
    sig_pos, # left px
    sig_vel, # left vx
    sig_f, # left fx
    sig_pos, # left py
    sig_vel, # left vy
    sig_f, # left fy
    sig_pos, # center cursor px
    sig_pos, # center cursor py
    sig_pos, # rh target px
    sig_pos, # rh target py
    sig_pos, # lh target px
    sig_pos, # lh target py
    sig_pos, # finaltarget py
]]).T

sensor_cov = 0.03
W1_cov =  np.diag(np.array([
    sensor_cov, # right px
    sensor_cov, # right vx
    sensor_cov, # right fx
    sensor_cov, # right py
    sensor_cov, # right vy
    sensor_cov, # right fy
    sensor_cov, # left px
    sensor_cov, # left vx
    sensor_cov, # left py
    sensor_cov, # left vy
    sensor_cov, # center cursor px
    sensor_cov, # center cursor py
    sensor_cov, # rh target px
    sensor_cov, # rh target py
    sensor_cov, # lh target px
    sensor_cov, # lh target py
    sensor_cov, # lh finaltarget py
]))
W2_cov =  np.diag(np.array([
    sensor_cov, # right px
    sensor_cov, # right vx
    sensor_cov, # right py
    sensor_cov, # right vy
    sensor_cov, # left px
    sensor_cov, # left vx
    sensor_cov, # left fx
    sensor_cov, # left py
    sensor_cov, # left vy
    sensor_cov, # left fy
    sensor_cov, # center cursor px
    sensor_cov, # center cursor py
    sensor_cov, # rh target px
    sensor_cov, # rh target py
    sensor_cov, # lh target px
    sensor_cov, # lh target py
    sensor_cov, # finaltarget py
]))

process_cov = 0.03
process_adj = 100 # Higher process noise on partner, trust sensory info more for them
V1_cov =  np.diag(np.array([
    process_cov, # right px
    process_cov, # right vx
    process_cov, # right fx
    process_cov, # right py
    process_cov, # right vy
    process_cov, # right fy
    process_cov*process_adj, # left px
    process_cov*process_adj, # left vx
    process_cov*process_adj, # left fx
    process_cov*process_adj, # left py
    process_cov*process_adj, # left vy
    process_cov*process_adj, # left fy
    process_cov, # center cursor px
    process_cov, # center cursor py
    process_cov, # rh target px
    process_cov, # rh target py
    process_cov, # lh target px
    process_cov, # lh target py
    process_cov, # finaltarget py
]))

V2_cov =  np.diag(np.array([
    process_cov*process_adj, # right px
    process_cov*process_adj, # right vx
    process_cov*process_adj, # right fx
    process_cov*process_adj, # right py
    process_cov*process_adj, # right vy
    process_cov*process_adj, # right fy
    process_cov, # left px
    process_cov, # left vx
    process_cov, # left fx
    process_cov, # left py
    process_cov, # left vy
    process_cov, # left fy
    process_cov, # center cursor px
    process_cov, # center cursor py
    process_cov, # rh target px
    process_cov, # rh target py
    process_cov, # lh target px
    process_cov, # lh target py
    process_cov, # finaltarget py
]))

help_percentages = [0.0, 0.0, 1.0, 0.5]
partner_knowledge = [False, True, True, True]