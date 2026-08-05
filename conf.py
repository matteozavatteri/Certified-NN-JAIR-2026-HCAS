InputSIZE = 3
OutputSIZE = 5

# inputs
rho   = 0
theta = 1
psi   = 2

# input bounds (normalized)
InputLowerBounds = [-0.21, -0.5, -0.5]
InputUpperBounds = [0.8, 0.5, 0.5]

# outputs
coc = 0
wl  = 1
wr  = 2
sl  = 3
sr  = 4

ActionsDifferentFromCOC = [wl, wr, sl, sr]
ActionsDifferentFromWL  = [coc, wr, sl, sr]
ActionsDifferentFromSR  = [coc, wl, wr, sl]
ActionsDifferentFromSL  = [coc, wl, wr, sr]

ActionsDifferentFromSL_SR = [coc, wl, wr]
ActionsDifferentFromCOC_WL = [wr, sl, sr]

COLORS = ['#482475', '#355f8d', '#21918c', '#44bf70', '#c6df26']

AMPLIFIER = 40


