import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import metrics
import os
import glob
import math
import pathlib

def lor(x, y):
    return [a + b for a, b in zip(x, y)]

def evaluate(trial, velo=False, proba=False, dyna=False, log=False, gstd=True, penalty='H', hidden=[32], lr=0.01, lag=5, max_iter=10000, lam_ridge=0):
    
    folder = trial + '-' + str(velo) + '-' + str(proba) + '-' + str(dyna) + '-' + str(log) + '-' + str(gstd) + '-' + str(max_iter) + '-' + str(hidden) + '-' + str(lag) + '-' + str(penalty) + '-' + str(lam_ridge) + '-' + str(lr)

    lag = int(folder.split('-')[-4])

    lams = [x[0].split('/')[-1] for x in os.walk('img/' + folder)][1:]
    lams = [float(x) for x in lams]

    mat1 = pd.read_csv('img/' + folder + "/1.0/preds.csv", index_col = 0)
    gt = mat1['y_true'].tolist()

    gc = []
    g = 0
    
    for lam in lams:
        gclam = []
        try:
            mat = torch.load('img/' + folder + "/" + str(lam) + '/lag.pt')
            mat = mat.detach()
            for l in range(lag):
                g = len(mat[:,:,l])
                gclam.append(mat[:,:,l].numpy().flatten())
            gc.append(gclam)
        except:
            pass


    ## produce matrices for each lag

    rgc = []

    for i in range(lag):
        rmats = []
        for j in range(len(gc)):
            nonzero = 0
            for val in gc[j][i]:
                if val > 1e-8:
                    nonzero += 1
            pnzero = nonzero / len(gc[j][i])

            if pnzero >= 0.01 and pnzero <= 0.95:
                rmats.append(gc[j][i])
        rgc.append(rmats)


    for l in range(lag):
        for i in range(len(rgc[l]) - 2, -1, -1):
            rgc[l][i] = lor(rgc[l][i], rgc[l][i + 1])

    pathlib.Path('eval/' + folder).mkdir(parents=True, exist_ok=True)
    
    for l in range(lag):
        for i in range(g):
            rgc[l][0][g*i + i] = 0

    for l in range(lag):

        np.savetxt('eval/' + folder + '/gc_' + str(lag - l) + '.csv', np.array(rgc[l][0]).reshape(100, 100), delimiter=",")

        fpr, tpr, threshold = metrics.roc_curve(gt, rgc[l][0])
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        plt.savefig('eval/' + folder + '/roc_' + str(lag - l) + '.png', bbox_inches='tight')
        plt.clf()

    for l in range(lag):
        precision, recall, threshold = metrics.precision_recall_curve(gt, rgc[l][0])
        prc = metrics.average_precision_score(gt, rgc[l][0])

        plt.title('Precision Recall Curve')
        plt.plot(recall, precision, 'b', label = 'AUC = %0.3f' % prc)
        plt.legend(loc = 'lower right')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('Precision')
        plt.xlabel('Recall')

        plt.savefig('eval/' + folder + '/prc_' + str(lag - l) + '.png', bbox_inches='tight')
        plt.clf()

    fgc = [0] * len(gt)

    for l in range(lag):
        fgc = lor(fgc, rgc[l][0])

    np.savetxt('eval/' + folder + '/gc_full.csv', np.array(fgc).reshape(100, 100), delimiter=",")

    fpr, tpr, threshold = metrics.roc_curve(gt, fgc)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig('eval/' + folder + '/roc_full.png', bbox_inches='tight')
    plt.clf()

    precision, recall, threshold = metrics.precision_recall_curve(gt, fgc)
    prc = metrics.average_precision_score(gt, fgc)

    plt.title('Precision Recall Curve')
    plt.plot(recall, precision, 'b', label = 'AUC = %0.3f' % prc)
    plt.legend(loc = 'lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')

    plt.savefig('eval/' + folder + '/prc_full.png', bbox_inches='tight')
    plt.clf()
    
trials = ['De-noised_100G_3T_300cPerT_dynamics_9_DS4',
          'De-noised_100G_4T_300cPerT_dynamics_10_DS5',
          'De-noised_100G_6T_300cPerT_dynamics_7_DS6',
          'De-noised_100G_7T_300cPerT_dynamics_11_DS7']


for trial in trials:
    for velo in [True]:
        for proba in [False, True]:
            for log in [False]:
                for gstd in [True]:
                    try:
                        evaluate(trial=trial, velo=velo, proba=proba, log=log, gstd=gstd)
                    except:
                        pass

'''
done = next(os.walk('img/'))[1]

for folder in done:
    if (not os.path.exists('eval/' + folder + '/roc_full.png')):
        try:
            evaluate(folder)
        except:
            pass
'''