import os 
import json

PROPERTIES = [1,2,3,4,5,6,7,8,9,10,12345610,123456810,123456910]

BRAINS = [(6,50)]
LEARNING_RATES = [1e-3]
SEEDS = [100]

STRATEGIES = [(50,20)]

SOFT = ['','-SA']
EPSILON = [1e-2,1e-3,1e-4]
DELTA   = [1e-2,1e-3,1e-4]

PRA = [0,1,2,3,4]
# time to loss of vertical separation
TAU = ['00','05','10','15','20','30','40','60']

key_template = (
    'pra{pra}-tau{tau}/'
    '{property}-'
    '{epsilon}-'
    '{delta}-'
    '{learning_rate}-'
    '{layers}x{neurons}-'
    '{counterexamples}-'
    '{sampling}-'
    '{seed}-L'
    '{soft}'
    )


KEYS = [key_template.format(
    pra=pra,
    tau=tau,
    property=p, 
    epsilon=epsilon,
    delta=delta, 
    learning_rate=learning_rate, 
    layers=layers,
    neurons=neurons,
    counterexamples=counterexamples, 
    sampling=sampling, 
    seed=seed, 
    soft=soft
    ) 
    for p in PROPERTIES 
    for pra in PRA 
    for tau in TAU 
    for epsilon in EPSILON 
    for delta in DELTA 
    for learning_rate in LEARNING_RATES 
    for (layers,neurons) in BRAINS 
    for (counterexamples,sampling) in STRATEGIES 
    for seed in SEEDS 
    for soft in SOFT 
    if delta > epsilon and (
        (p in [1,2,3,4,5,6,10,12345610]) or 
        (p == 7 and tau == '60') or 
        (p == 8 and pra == 1 and tau == '60') or 
        (p == 9 and pra == 2) or
        (p == 123467810 and pra == 1 and tau == '60') or
        (p == 123456810 and pra == 1 and tau == '60') or
        (p == 123456910 and pra == 2 and tau == '60') 
        )]


def load_experiment_data():
    data = dict()
    for key in KEYS:
        path = 'experiments/{}'.format(key)
        if os.path.exists(path):
            with open('{}/stats.json'.format(path), "r") as f:
                data[key] = json.load(f)
    return data

if __name__ == '__main__':
    data = load_experiment_data()
    total = len(data)
    missing = [k for k in KEYS if k not in data]
    timeout = [k for k in data.keys() if data[k]['TotalTIME'] == None]

    print('total={}'.format(total))
    print('missing={}'.format(missing))
    print('timeout={}'.format(timeout))
