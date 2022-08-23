import sys
import os

# Please select dataset(RBP-24/RBP-120) and experiment(sequence/vivo/vitro)."
params = sys.argv

if len(params) != 3:
    print("Please enter two parameters for the dataset and the experiment.")
else:
    if (sys.argv[1] in ['RBP-24','RBP-120']) & (sys.argv[2] in ['sequence','vivo','vitro']):
        dataset = sys.argv[1]
        experiment = sys.argv[2]
    else:
        print("Parameter input is incorrectly and the default settings are performed.")
        dataset = 'RBP-120'
        experiment = 'sequence'

    if (dataset == 'RBP-24') & (experiment != 'sequence'):
        print("Please reselect the experiment.")
    if (dataset == 'RBP-24') & (experiment == 'sequence'):    
        os.system('python3 RBP-24/trainer.py')
    if (dataset == 'RBP-120') & (experiment == 'sequence'):
        os.system('python3 RBP-120/sequence-only/Rbp120trainer.py')
    if (dataset == 'RBP-120') & (experiment == 'vivo'):
        os.system('python3 RBP-120/sequence+vivo/Rbp120trainerVivo.py')
    if (dataset == 'RBP-120') & (experiment == 'vitro'):
        os.system('python3 RBP-120/sequence+vitro/Rbp120trainerVitro.py')
