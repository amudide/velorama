#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models.utils as UTILS
import os, glob, sys
import sklearn.metrics



def get_beeline_edge_scores(dirname, method, is_synthetic_data = True):
    if method == 'VeloNet':
        fname1 = f'{dirname}/velonet_results.csv'
        fname2 = f'{dirname}' 

        print('Flag 654.30 ', fname1, fname2)
        if not (os.path.exists(fname1) or os.path.exists(fname2)):
            return None
        
        if os.path.exists(fname1):
            df1 =  pd.read_csv(fname1)
        elif os.path.exists(fname2):
            df1 =  pd.read_csv(fname2)
            
        if df1.shape[0] > 5:
            df1.columns = ['g1', 'g2', 'score']
            dfscores = df1
            return dfscores
        else:
            return None
    
    dfscores = None
    if os.path.exists(f'{dirname}/rankedEdges.csv'):
        df1 = pd.read_csv(f'{dirname}/rankedEdges.csv', delimiter='\t')
        if df1.shape[0] > 5:
            df1.columns = ['g1','g2','score']
            dfscores = df1

    if (not is_synthetic_data) and method=='GENIE3' and os.path.exists(f'{dirname}/outFile.txt'):
        df1 = pd.read_csv(f'{dirname}/outFile.txt', delimiter='\t')
        if df1.shape[0] > 5:
            df1.columns = ['g1', 'g2', 'score']
            dfscores = df1
        
    if not is_synthetic_data: return dfscores
            
    if dfscores is None:
        outf = glob.glob(f'{dirname}/outFile*')[0]
        have_header = method in ['SCODE', 'GRNBOOST2']
        if have_header:
            df1 = pd.read_csv(outf, delimiter='\t')
        else:
            df1 = pd.read_csv(outf, delimiter='\t', header=None)
        if df1.shape[0] > 5:
            df1.columns = ['g1', 'g2', 'score']
            dfscores = df1

    if dfscores is None: return None

    if is_synthetic_data:
        for c in ['g1','g2']:
            if dfscores[c].dtype == 'object':
                dfscores[c] =  [int(s[1:]) for s in dfscores[c]]
            else:
                dfscores[c] = dfscores[c].astype(int)
    return dfscores




        
def eval_synthetic_results():
    methods = ['PIDC','GRNBOOST2','SCODE','SINGE', 'SINCERITIES', 'SCRIBE'] # GENIE3 didn't run on synthetic data
    
    DIR1 = '/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/beeline-murali/code/Beeline/outputs/SERGIO/'
    datasets = [ DIR1 + 'De-noised_100G_3T_300cPerT_dynamics_9_DS4',
                 DIR1 + 'De-noised_100G_6T_300cPerT_dynamics_7_DS6',
                 DIR1 + 'De-noised_100G_4T_300cPerT_dynamics_10_DS5',
                 DIR1 + 'De-noised_100G_7T_300cPerT_dynamics_11_DS7']

    DIR2 = "/afs/csail.mit.edu/u/a/amudide/gc/data_sets/"

    results = {}
    
    for dset in datasets: #[:1]:
        basedir = dset.split('/')[-1]
        results[basedir] = []

        print("Flag 223.20 ", dset, basedir) #, results)
        
        true_edges = np.loadtxt(f'{DIR2}/{basedir}/gt_GRN.csv', delimiter=",").astype(int)
        n1 = 1+max( true_edges[:,0].flatten().max(), true_edges[:,1].flatten().max())
        print("Flag 223.25 ", true_edges.shape, n1)
        
        for method in methods:
            print("Flag 223.30 ", method)
            
            dfscores = get_beeline_edge_scores(f'{dset}/{method}/', method, is_synthetic_data=True)
            if dfscores is None:
                print("Flag 223.31 ", method, " had no results")
                continue
            
            
            l1 = sorted(list(set(dfscores["g1"].tolist() + dfscores["g2"].tolist())))
            n = max(n1, len(l1))

            print("Flag 223.32 ", dfscores.shape, n1, len(l1), n)
            
            gene2idx = {g:i for i,g in enumerate(l1)}

            ytrue = np.zeros((n,n))
            ypred = ytrue.copy()

            for e in true_edges:
                ytrue[e[0],e[1]] = 1
                
            for g1,g2,score in zip(dfscores['g1'], dfscores['g2'], dfscores['score']):
                ypred[ gene2idx[g1], gene2idx[g2]] = score

            ytrue1 = ytrue.flatten()
            ypred1 = ypred.flatten()

            auroc = sklearn.metrics.roc_auc_score( ytrue1, ypred1)
            aupr = sklearn.metrics.average_precision_score( ytrue1, ypred1)

            results[basedir].append((auroc, aupr, method))

    L = []
    for dataset, vals in results.items():
        for auroc , aupr, method in vals:
            L.append([dataset, method, auroc, aupr])
    dfresults = pd.DataFrame(L, columns = 'dataset,method,auroc,aupr'.split(','))

    dfresults.to_csv(sys.stdout, index=False)
    
    return results


        
def eval_scrnaseq_results():
    methods = ['GENIE3', 'PIDC','GRNBOOST2','SCODE','SINGE', 'SINCERITIES', 'SCRIBE', 'VeloNet'] # GENIE3 didn't run on synthetic data
    
    DIR1 = '/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/beeline-murali/code/Beeline/outputs/scRNA-Seq/'
    datasets = [  DIR1 + "s4_hESC_500-50",
                  DIR1 + "s4_hHep_500-50",
                  DIR1 + "s4_mDC_500-50",
                  DIR1 + "s4_mESC_500-50",
                  DIR1 + "s4_mHSC-E_500-50",
                  DIR1 + "s4_mHSC-GM_500-50",
                  DIR1 + "s4_mHSC-L_500-50",
    ]

    tissue2validation = {
        "hESC": ("human","hESC-ChIP-seq-network.csv","Non-specific-ChIP-seq-network.csv"),
        "hHep": ("human","HepG2-ChIP-seq-network.csv","Non-specific-ChIP-seq-network.csv"),
        "mDC": ("mouse","mDC-ChIP-seq-network.csv","Non-Specific-ChIP-seq-network.csv"),
        "mESC": ("mouse","mESC-ChIP-seq-network.csv","Non-Specific-ChIP-seq-network.csv"),
        "mHSC-E": ("mouse","mHSC-ChIP-seq-network.csv","Non-Specific-ChIP-seq-network.csv"),
        "mHSC-GM": ("mouse","mHSC-ChIP-seq-network.csv","Non-Specific-ChIP-seq-network.csv"),
        "mHSC-L": ("mouse","mHSC-ChIP-seq-network.csv","Non-Specific-ChIP-seq-network.csv"),
    }
    
    DIR2 = "/afs/csail.mit.edu/u/a/amudide/gc/data_sets/"
    DIR3 = "/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/beeline-murali/raw/Networks/"
    DIR4 = "/afs/csail.mit.edu/u/r/rsingh/work/perrimon-sc/data/beeline-murali/code/Beeline/inputs/scRNA-Seq/"
    
    results = {}
    
    for dset in datasets: #[:3]: 
        basedir = dset.split('/')[-1]
        tissue = basedir.split('_')[1]
        results[tissue] = []

        print("Flag 223.20 ", dset, tissue) #, results)

        f_read_genelist = lambda fname: list(set([s.strip() for s in open(fname,'rt') if s.strip()!=""]))
        tfgenes = f_read_genelist(DIR4 + basedir + "/tf_genes.txt")
        nontfgenes = f_read_genelist(DIR4 + basedir + "/nontf_genes.txt")

        tf_id2idx = {g:i for i,g in enumerate(tfgenes)}
        nontf_id2idx = {g:i for i,g in enumerate(nontfgenes)}
        n1, n2 = len(tf_id2idx), len(nontf_id2idx)
        
        sp, chipseq1_file, chipseq2_file = tissue2validation[tissue]

        for ii, fname in enumerate([chipseq1_file, chipseq2_file]):
            df1 = pd.read_csv(DIR3 + sp + "/" + fname)
            if 'Score' not in df1.columns:
                df1["Score"] = 1.0
                
            df1['Gene1'] = df1['Gene1'].str.upper()
            df1['Gene2'] = df1['Gene2'].str.upper()
            

            df1 = df1.loc[ df1['Gene1'].isin(tfgenes),:].reset_index(drop=True)
            df1 = df1.loc[ df1['Gene2'].isin(nontfgenes),:].reset_index(drop=True)

            ytrue = np.zeros((n1,n2))
            for g1,g2 in zip(df1['Gene1'], df1['Gene2']):
                ytrue[ tf_id2idx[g1], nontf_id2idx[g2] ] = 1                    

            print("Flag 223.25 ", ytrue.shape, (ytrue > 0).sum(), (ytrue > 0).mean(), n1, n2)
        
            for method in methods:
                print("Flag 223.30 ", dset, method)

                if method != 'VeloNet':
                    dfscores = get_beeline_edge_scores(f'{dset}/{method}/', method, is_synthetic_data=False)
                else:
                    #dfscores = get_beeline_edge_scores(f'{dset}/', method, is_synthetic_data=False)
                    #dfscores = get_beeline_edge_scores(f'/data/cb/alexwu/tf_lag/beeline_results/{basedir}.results2.csv', method, is_synthetic_data=False)
                    #dfscores = get_beeline_edge_scores(f'/data/cb/alexwu/tf_lag/beeline_results/{basedir}.results.binarize.csv', method, is_synthetic_data=False)
                    dfscores = get_beeline_edge_scores(f'/data/cb/alexwu/tf_lag/beeline_results/{basedir}.results.mean.0mean_1sd.h32.new.csv', method, is_synthetic_data=False)

                if dfscores is None:
                    print("Flag 223.31 ", method, " had no results for ", tissue)
                    continue


                print("Flag 223.32 ", dfscores.shape, n1, n2)
                dfscores = dfscores.loc[ dfscores['g1'].isin(tfgenes),:].reset_index(drop=True)
                dfscores = dfscores.loc[ dfscores['g2'].isin(nontfgenes),:].reset_index(drop=True)

                print("Flag 223.33 ", dfscores.shape)

                ypred = np.zeros((n1,n2))

                for g1,g2,score in zip(dfscores['g1'], dfscores['g2'], dfscores['score']):
                    ypred[ tf_id2idx[g1], nontf_id2idx[g2]] = score

                ytrue1 = ytrue.flatten()
                ypred1 = ypred.flatten()

                auroc = sklearn.metrics.roc_auc_score( ytrue1, ypred1)
                aupr = sklearn.metrics.average_precision_score( ytrue1, ypred1)

                chipseq_type = "tissue-specific" if ii==0 else "non-tissue-specific"
                results[tissue].append((auroc, aupr, method, chipseq_type))

                    
    L = []
    for tissue, vals in results.items():
        for auroc , aupr, method, chipseq_type in vals:
            L.append([tissue, chipseq_type, method, auroc, aupr])
    dfresults = pd.DataFrame(L, columns = 'tissue,chipseq_type,method,auroc,aupr'.split(','))

    dfresults.to_csv(sys.stdout, index=False)
    
    return results


        
#####################################

if __name__ == "__main__":
    #convert_sergio_synthetic_to_beeline()
    #prep_beeline_scrnaseq_data()
    #eval_synthetic_results()
    eval_scrnaseq_results()
