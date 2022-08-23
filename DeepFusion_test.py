import sys
import os

# Please select dataset(RBP-24/RBP-120).
params = sys.argv

if len(params) != 2:
    print("Please enter a dataset parameter.")
else:
    if (sys.argv[1] == 'RBP-24') or (sys.argv[1] == 'RBP-120'):
        dataset = sys.argv[1]
    else:
        print("Parameter is incorrectly and the default settings are performed.")
        dataset = 'RBP-120'

    if dataset == 'RBP-24':
        os.system('python3 RBP-24/testner.py')
    if dataset == 'RBP-120':
        os.system('python3 RBP-120/sequence-only/Rbp120testner.py')
