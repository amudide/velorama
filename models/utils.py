import numpy as np
import os
import torch
from scipy.stats import f
from scipy.sparse import csr_matrix
import scanpy as sc
import scanpy.external as sce
from anndata import AnnData
import cellrank as cr
from cellrank.tl.kernels import VelocityKernel
import pandas as pd


def construct_dag(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt'):
	
	"""Constructs the adjacency matrix for a DAG.
	Parameters
	----------
	joint_feature_embeddings: 'numpy.ndarray' (default: None)
		Matrix of low dimensional embeddings with rows corresponding
		to observations and columns corresponding to feature embeddings
		for constructing a DAG if a custom DAG is not provided.
	iroot: 'int' (default: None)
		Index of root cell for inferring pseudotime for constructing a DAG 
		if a custom DAG is not provided.
	n_neighbors: 'int' (default: 15)
		Number of nearest neighbors to use in constructing a k-nearest
		neighbor graph for constructing a DAG if a custom DAG is not provided.
	pseudotime_algo: {'dpt','palantir'} 
		Pseudotime algorithm to use for constructing a DAG if a custom DAG 
		is not provided. 'dpt' and 'palantir' perform the diffusion pseudotime
		(Haghverdi et al., 2016) and Palantir (Setty et al., 2019) algorithms, 
		respectively.
	"""

	pseudotime,knn_graph = infer_knngraph_pseudotime(joint_feature_embeddings,iroot,
		n_neighbors=n_neighbors,pseudotime_algo=pseudotime_algo)
	dag_adjacency_matrix = dag_orient_edges(knn_graph,pseudotime)

	return dag_adjacency_matrix

def infer_knngraph_pseudotime(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt'):

	adata = AnnData(joint_feature_embeddings)
	adata.obsm['X_joint'] = joint_feature_embeddings
	adata.uns['iroot'] = iroot

	if pseudotime_algo == 'dpt':
		sc.pp.neighbors(adata,use_rep='X_joint',n_neighbors=n_neighbors)
		sc.tl.dpt(adata)
		adata.obs['pseudotime'] = adata.obs['dpt_pseudotime'].values
		knn_graph = adata.obsp['distances'].astype(bool).astype(float)
	elif pseudotime_algo == 'palantir':
		sc.pp.neighbors(adata,use_rep='X_joint',n_neighbors=n_neighbors)
		sce.tl.palantir(adata, knn=n_neighbors,use_adjacency_matrix=True,
			distances_key='distances')
		pr_res = sce.tl.palantir_results(adata,
			early_cell=adata.obs.index.values[adata.uns['iroot']],
			ms_data='X_palantir_multiscale')
		adata.obs['pseudotime'] = pr_res.pseudotime
		knn_graph = adata.obsp['distances'].astype(bool).astype(float)

	return adata.obs['pseudotime'].values,knn_graph

def dag_orient_edges(adjacency_matrix,pseudotime):

	A = adjacency_matrix.astype(bool).astype(float)
	print(pseudotime[:,None].shape)
	print(pseudotime.shape)
	D = -1*np.sign(pseudotime[:,None] - pseudotime).T
	D = (D == 1).astype(float)
	D = (A.toarray()*D).astype(bool).astype(float)

	return D

def construct_S(D):
    
    #return D.T
    
    S = D.clone()
    D_sum = D.sum(0)
    D_sum[D_sum == 0] = 1
    
    S = (S/D_sum)
    S = S.T
    
    return S

def load_multiome_data(data_dir,dataset,sampling=None,preprocess=True):

	if sampling == 'geosketch':
		atac_adata = sc.read(os.path.join(data_dir,
			'{}.atac.sketch.h5ad'.format(dataset)))
		rna_adata = sc.read(os.path.join(data_dir,
			'{}.rna.sketch.h5ad'.format(dataset)))
	elif sampling == 'uniform':
		atac_adata = sc.read(os.path.join(data_dir,
			'{}.atac.uniform.h5ad'.format(dataset)))
		rna_adata = sc.read(os.path.join(data_dir,
			'{}.rna.uniform.h5ad'.format(dataset)))
	else:
		atac_adata = sc.read(os.path.join(data_dir,
			'{}.atac.h5ad'.format(dataset)))
		rna_adata = sc.read(os.path.join(data_dir,
			'{}.rna.h5ad'.format(dataset)))

	if preprocess:

		# scale by maximum 
		# (rna already normalized by library size + log-transformed)
		X_max = rna_adata.X.max(0).toarray().squeeze()
		X_max[X_max == 0] = 1
		rna_adata.X = csr_matrix(rna_adata.X / X_max)

		# atac: normalize library size + log transformation
		sc.pp.normalize_total(atac_adata,target_sum=1e4)
		sc.pp.log1p(atac_adata)

	return rna_adata,atac_adata

def seq2dag(N):
    A = torch.zeros(N, N)
    for i in range(N - 1):
        A[i][i + 1] = 1
    return A



def guess_iroot(m, stemcell_frac_thresh=0.05): # num of genes expressed is a proxy for stemness.

        assert type(m) == np.ndarray and len(m.shape)==2
        
        m1 = (m>1e-8).sum(axis=1).flatten()
        m1_topK = np.quantile(m1, 1-stemcell_frac_thresh)
        idx_topK = (m1 >= m1_topK) #likely stem cells
        mean_exp_topK = m[idx_topK, :].mean(axis=0).flatten()
        
        #print(m1.shape, m1_topK, idx_topK.shape, idx_topK.sum(), mean_exp_topK.shape)
        m1_dist = ((m - mean_exp_topK[None, :])**2).sum(axis=1)
        m1_dist[~idx_topK] = 1e9 # don't count non-stem cells

        iroot  = np.argmin(m1_dist) # closest to center of stem cell cluster
        return iroot



def produce_beeline_inputs(transcript_counts_file, outdir):
        X = pd.read_csv(transcript_counts_file, index_col=0, header=None, skiprows=[0])

        barcodes = X.columns.tolist()
        genes = X.index.tolist()
        
        X = X.to_numpy()
        X = np.transpose(X)
        adata = AnnData(X, dtype=np.float32)

        print('Flag 592.20 ', adata.shape, len(barcodes), len(genes))

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver='arpack')

        #typically we'd call guess_iroot(adata.X) but this is synthetic data and ok to assume first cell was near start of differentiation traj
        iroot = 0 # 
        
        dpt, _  = infer_knngraph_pseudotime(adata.X, iroot)
        adata.obs['dpt'] = dpt

        print('Flag 592.30 ', iroot, adata.obs['dpt'].describe())
              
        df = pd.DataFrame(adata.X).T
        print('Flag 592.32 ', df.shape)
        df.columns = barcodes
        print('Flag 592.34 ', df.shape)
        df.index = genes
        print('Flag 592.36 ', df.shape)

        df.to_csv(f'{outdir}/ExpressionData.csv', index=True)

        df1 = adata.obs['dpt'].to_frame()
        df1.index.name = ''
        df1.to_csv(f'{outdir}/PseudoTime.csv')


        
