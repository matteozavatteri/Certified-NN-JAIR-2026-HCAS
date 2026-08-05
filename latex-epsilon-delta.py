import os 

import experiments
import statistics

data = experiments.load_experiment_data()

def textEpsilonDelta(raw):
    exp = 2 if raw == 1e-2 else 3 if raw == 1e-3 else 4 if raw == 1e-4 else False
    return '$10^{{-{exp}}}$'.format(exp=exp)

with open('table-epsilon-delta.tex', 'w') as f:
    f.write('\\scalebox{0.5}{\\begin{tabular}{|*{17}{c|}}\n')
    f.write('\\hline\n')
    
    f.write(
        ('\\multicolumn{3}{|l|}{} & '
        '\\multicolumn{14}{l|}{{\\bf Average value $\\pm$ standard deviation}} \\\\\n'))    
    
    f.write('\\cline{4-17}\n')
    
    f.write(
            ('\\multicolumn{3}{|l|}{} &'
             '\\multicolumn{2}{l|}{{\\bf Whole process}} &'
             '\\multicolumn{12}{l|}{{\\bf CEGIS iteration}} \\\\\n')
    )
    
    f.write('\\hline\n')
    
    f.write(
        (' & '
        '{\\bf \\makecell{$\\varepsilon$}} & '
        '{\\bf \\makecell{$\\Delta$}} & ' 
        '{\\bf \\makecell{TO}} & '
        '{\\bf \\makecell{CEGIS\\\\iterations}} & '
        '{\\bf \\makecell{Total\\\\time (s)}} & '
        '{\\bf \\makecell{Training\\\\time (s)}} & '
        '{\\bf \\makecell{Opportun. \\\\search time (s)}} & '
        '{\\bf \\makecell{FV \\\\ time (s)}} & '
        '{\\bf \\makecell{Repair \\\\ time (s)}} & '
        '{\\bf \\makecell{Initial loss \\\\ on valid}} & '
        '{\\bf \\makecell{Final loss \\\\ on valid}} & '
        '{\\bf \\makecell{Initial loss \\\\ on test}} & '
        '{\\bf \\makecell{Final loss \\\\ on test}} & '
        '{\\bf \\makecell{CounterEX \\\\ (opportun.)}} & '
        '{\\bf \\makecell{FV \\\\ calls}} & '
        '{\\bf \\makecell{New \\\\ pairs}}  \\\\\n'))
    f.write('\\hline\n')

    EPSILON_DELTA = [(e,d) for e in experiments.EPSILON for d in experiments.DELTA if d > e]

    for P in experiments.PROPERTIES:
        if P <= 10:
            textP = '$P_{{{}}}$'.format(P)
        elif P == 12345610:
            textP = '$\\mathcal{{P}}_1$'
        elif P == 123456810:
            textP = '$\\mathcal{{P}}_2$'
        elif P == 123456910:
            textP = '$\\mathcal{{P}}_3$'
        else:
            assert False

        f.write("\\multirow{{{}}}{{*}}{{{}}} ".format(2*3,textP))
        for (epsilon,delta) in EPSILON_DELTA:
            f.write("& \\multirow{{{}}}{{*}}{{{}}} ".format(2,textEpsilonDelta(epsilon)))
            f.write("& \\multirow{{{}}}{{*}}{{{}}} ".format(2,textEpsilonDelta(delta)))
            first = True
            for soft in experiments.SOFT:
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
                for learning_rate in experiments.LEARNING_RATES 
                for (layers,neurons) in experiments.BRAINS 
                for (counterexamples,sampling) in experiments.STRATEGIES 
                for seed in experiments.SEEDS  
                if delta > epsilon and (
                    (P in [1,2,3,4,5,6,10,12345610]) or 
                    (P == 7 and tau == '60') or 
                    (P == 8 and pra == 1 and tau == '60') or 
                    (P == 9 and pra == 2) or
                    (P == 123467810 and pra == 1 and tau == '60') or
                    (P == 123456810 and pra == 1 and tau == '60') or
                    (P == 123456910 and pra == 2 and tau == '60') 
                    )]
                
                ROW = dict()

                # remove all experiments that hit the timeout
                TERMINATED = [k for k in ALL if k in data.keys() and data[k]['TotalTIME'] is not None]
                     

                TIMEOUT = [k for k in ALL if k in data.keys() and data[k]['TotalTIME'] is None]
                if len(TIMEOUT) > 0:
                    print('TIMEOUT={}'.format(TIMEOUT))


                MISSING = [k for k in ALL if k not in data.keys()]
                if len(MISSING) > 0:
                    print('MISSING={}'.format(MISSING))

                TO = len(ALL) - len(TERMINATED)

                if not first:
                    f.write(' & & ')
                first = False

                if len(TERMINATED) == 0:
                    f.write(
                    (' & '
                    '{TO} & '
                    '- & '
                    '- & '
                    '- & '
                    '- & '
                    '- & ' 
                    '- & ' 
                    '- & ' 
                    '- & ' 
                    '- & ' 
                    '- & ' 
                    '- & ' 
                    '- & ' 
                    '- \\\\\n'.format(
                        TO=TO)))
                    continue

                mean = statistics.mean(data[k]['CEGISiterations'] for k in TERMINATED)
                stdev = statistics.stdev(data[k]['CEGISiterations'] for k in TERMINATED) if len(TERMINATED)>1 else 0
                ROW['CEGISiterations'] = '${}\\pm{}$'.format(round(mean), round(stdev))

                mean = statistics.mean(data[k]['TotalTIME'] for k in TERMINATED)
                stdev = statistics.stdev(data[k]['TotalTIME'] for k in TERMINATED) if len(TERMINATED)>1 else 0
                ROW['TotalTIME'] = '${}\\pm{}$'.format(round(mean ,2), round(stdev ,2))
                
                for metric in ['TrainingTIME', 'OpportunisticSeachTIME', 'FormalVerificationTIME', 'RepairTIME']:
                    tmp = list(data[k][metric] for k in TERMINATED)
                    tmp = [x for y in tmp for x in y]
                    mean = statistics.mean(tmp)
                    stdev = statistics.stdev(tmp) if len(TERMINATED)>1 else 0
                    ROW[metric]  = '${}\\pm{}$'.format(round(mean,2), round(stdev,2))

                for metric in ['OpportunisticCEX', 'NewPairs']:
                    tmp = list(data[k][metric] for k in TERMINATED)
                    tmp = [x for y in tmp for x in y]
                    mean = statistics.mean(tmp)
                    stdev = statistics.stdev(tmp) if len(TERMINATED)>1 else 0
                    ROW[metric]  = '${}\\pm{}$'.format(round(mean), round(stdev))

                tmp = [data[k]['OpportunisticCEX'].count(0) for k in TERMINATED]
                mean = statistics.mean(tmp)
                stdev = statistics.stdev(tmp) if len(TERMINATED)>1 else 0
                ROW['CallsToFV']  = '${}\\pm{}$'.format(round(mean), round(stdev))

                for metric in ['LossOnValidation', 'LossOnTest']:
                    mean = statistics.mean(data[k][metric][0]  for k in TERMINATED)
                    stdev = statistics.stdev(data[k][metric][0]  for k in TERMINATED) if len(TERMINATED)>1 else 0
                    ROW['Initial' + metric] = '${}\\pm{}$'.format(round(mean,3), round(stdev,3))

                    mean = statistics.mean(data[k][metric][-1]  for k in TERMINATED)
                    stdev = statistics.stdev(data[k][metric][-1]  for k in TERMINATED) if len(TERMINATED)>1 else 0
                    ROW['Final' + metric] = '${}\\pm{}$'.format(round(mean,3), round(stdev,3))

                COLOR = '\\cellcolor{lightgray!50}' if soft == '-SA' else ''

                

                f.write(
                    (' & '
                    '{COLOR}{TO} & '
                    '{COLOR}{CEGISiterations} & '
                    '{COLOR}{TotalTIME} & '
                    '{COLOR}{TrainingTIME} & '
                    '{COLOR}{OpportunisticSeachTIME} & ' 
                    '{COLOR}{FormalVerificationTIME} & '
                    '{COLOR}{RepairTIME} & ' 
                    '{COLOR}{InitialLossOnValidation} & ' 
                    '{COLOR}{FinalLossOnValidation} & ' 
                    '{COLOR}{InitialLossOnTest} & ' 
                    '{COLOR}{FinalLossOnTest} & ' 
                    '{COLOR}{OpportunisticCEX} & ' 
                    '{COLOR}{CallsToFV} & ' 
                    '{COLOR}{NewPairs} \\\\\n'.format(
                        COLOR=COLOR,
                        TO=TO,
                        CEGISiterations=ROW['CEGISiterations'],
                        TotalTIME=ROW['TotalTIME'],
                        TrainingTIME=ROW['TrainingTIME'],
                        OpportunisticSeachTIME=ROW['OpportunisticSeachTIME'],
                        FormalVerificationTIME=ROW['FormalVerificationTIME'],
                        RepairTIME=ROW['RepairTIME'],
                        InitialLossOnValidation=ROW['InitialLossOnValidation'],
                        FinalLossOnValidation=ROW['FinalLossOnValidation'],
                        InitialLossOnTest=ROW['InitialLossOnTest'],
                        FinalLossOnTest=ROW['FinalLossOnTest'],
                        OpportunisticCEX=ROW['OpportunisticCEX'],
                        CallsToFV=ROW['CallsToFV'],
                        NewPairs=ROW['NewPairs'])
                    )
                )
            f.write('\\cline{2-17}\n')
        f.write('\\hline\n')

    f.write('\n\\end{tabular}}')
