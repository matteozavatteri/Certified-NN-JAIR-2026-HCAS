import experiments
import statistics
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

data = experiments.load_experiment_data()
plot_rows = []

for P in experiments.PROPERTIES:
    fv_values = []
    repair_values = []

    ALL = [experiments.key_template.format(
                pra=pra,
                tau=tau,
                property=P, 
                epsilon=epsilon, 
                delta=delta,
                learning_rate=learning_rate, 
                layers=layers,
                neurons=neurons,
                counterexamples=counterexamples, 
                sampling=sampling, 
                seed=seed, 
                soft=soft
                ) \
                for pra in experiments.PRA 
                for tau in experiments.TAU 
                for epsilon in experiments.EPSILON
                for delta in experiments.DELTA
                for learning_rate in experiments.LEARNING_RATES 
                for (layers,neurons) in experiments.BRAINS 
                for (counterexamples,sampling) in experiments.STRATEGIES 
                for seed in experiments.SEEDS  
                for soft in experiments.SOFT
                if delta > epsilon and (
                    (P in [1,2,3,4,5,6,10,12345610]) or 
                    (P == 7 and tau == '60') or 
                    (P == 8 and pra == 1 and tau == '60') or 
                    (P == 9 and pra == 2) or
                    (P == 123467810 and pra == 1 and tau == '60') or
                    (P == 123456810 and pra == 1 and tau == '60') or
                    (P == 123456910 and pra == 2 and tau == '60') 
                    )]

    TERMINATED = [k for k in ALL if data[k]['TotalTIME'] is not None]
    tmp = list(data[k]['FormalVerificationTIME'] for k in TERMINATED)
    tmp = [x for y in tmp for x in y]
    FV = round(statistics.mean(tmp),2)

    tmp = list(data[k]['RepairTIME'] for k in TERMINATED)
    tmp = [x for y in tmp for x in y]
    R = round(statistics.mean(tmp),2)
    
    plot_rows.append({
        'P': P,
        'FV': FV,
        'R': R,
    })

labels = ['$P_{{{}}}$'.format(r['P']) for r in plot_rows]
labels[-3] = '$\\mathcal{P}_1$'
labels[-2] = '$\\mathcal{P}_2$'
labels[-1] = '$\\mathcal{P}_3$'

fv_means = [r['FV'] for r in plot_rows]
repair_means = [r['R'] for r in plot_rows]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))

ax.bar(x - width / 2, fv_means, width, label='Formal Verification time')
ax.bar(x + width / 2, repair_means, width, label='Repair time')

ax.set_ylabel('Time (s)')
ax.set_xlabel('Property')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('img/fv-vs-rep-HCAS.pdf')
plt.close()