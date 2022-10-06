#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models.utils as UTILS
import os


DIR1='/afs/csail.mit.edu/u/a/amudide/gc/data_sets/'
sergiofiles = [ DIR1 + 'De-noised_100G_3T_300cPerT_dynamics_9_DS4/simulated_noNoise_T_0.csv',
                DIR1 + 'De-noised_100G_6T_300cPerT_dynamics_7_DS6/T.csv',
                DIR1 + 'De-noised_100G_4T_300cPerT_dynamics_10_DS5/simulated_noNoise_T_0.csv',
                DIR1 + 'De-noised_100G_7T_300cPerT_dynamics_11_DS7/simulated_noNoise_T_0.csv']

for tfile in sergiofiles:
    DIR2 = '/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/beeline-murali/code/Beeline/inputs/SERGIO'
    outdir = '%s/%s' %( DIR2, tfile.split('/')[-2])
    os.system(f'mkdir -p {outdir}')
    print('Flag 592.30 converting file ', tfile)
    UTILS.produce_beeline_inputs(tfile, outdir)

