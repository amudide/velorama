#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models.utils as UTILS
import os


def convert_sergio_synthetic_to_beeline():
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



def prep_beeline_scrnaseq_data(ngenes=500, ntfs=50):
    tissues = 'hESC,hHep,mDC,mESC,mHSC-E,mHSC-GM,mHSC-L'.split(',')
    DIR2 = '/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/beeline-murali/code/Beeline/inputs/scRNA-Seq'

    for t in tissues:
        dfe = pd.read_csv(f"{DIR2}/{t}/ExpressionData.csv", index_col=0)
        dfe.index = [s.upper() for s in dfe.index]
        dfp = pd.read_csv(f"{DIR2}/{t}/PseudoTime.csv", index_col=0)
        dfo = pd.read_csv(f"{DIR2}/{t}/GeneOrdering.csv", index_col=0)
        dfo.index = [s.upper() for s in dfo.index]
        
        dfo.index.name = 'gene'
        dfo = dfo.reset_index()
        dfo = dfo.sort_values('VGAMpValue').reset_index(drop=True)

                
        sp = "human" if t[0]=='h' else "mouse"
        tfs = pd.read_csv(f"/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/beeline-murali/raw/{sp}-tfs.csv")

        print("Flag 208.40 ", t, dfe.shape, dfp.shape, dfo.shape, sp, tfs.shape, dfp.columns, dfo.columns)

        tf_genes = dfo.loc[ dfo['gene'].isin(tfs['TF']), 'gene'].values[:ntfs].tolist()
        nontf_genes = dfo.loc[ ~dfo['gene'].isin(tfs['TF']), 'gene'].values[:ngenes].tolist()
        
        print("Flag 208.50 ", len(tf_genes), len(nontf_genes), tf_genes[:5], nontf_genes[:5])
        allgenes = tf_genes + nontf_genes
        dfe1 = dfe.loc[ allgenes, :]

        outdir = f'{DIR2}/s4_{t}_{ngenes}-{ntfs}'
        os.system(f'mkdir -p {outdir}')
        dfe1.to_csv(f'{outdir}/ExpressionData.csv')
        dfp.to_csv(f'{outdir}/PseudoTime.csv')
        np.savetxt(f'{outdir}/nontf_genes.txt', nontf_genes, fmt='%s')
        np.savetxt(f'{outdir}/tf_genes.txt', tf_genes, fmt='%s')
        
#####################################

if __name__ == "__main__":
    #convert_sergio_synthetic_to_beeline()
    prep_beeline_scrnaseq_data()
