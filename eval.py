import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import metrics
import os
import glob
import pathlib

folder = 'De-noised_100G_6T_300cPerT_dynamics_7_DS6-False-10000-[100]-5-H-0-0.0001' ## change this line to evaluate other runs

lag = int(folder.split('-')[-4])

lams = np.logspace(-3.0, 3.0, num=39).tolist()

mat1 = pd.read_csv('img/' + folder + "/1.0/preds.csv", index_col = 0)
gt = mat1['y_true'].tolist()

gc = []

for lam in lams:
    gclam = []
    mat = torch.load('img/' + folder + "/" + str(lam) + '/lag.pt')
    mat = mat.detach()
    for l in range(lag):
        gclam.append(mat[:,:,l].numpy().flatten())
    gc.append(gclam)


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


def lor(x, y):
    return [a + b for a, b in zip(x, y)]

for l in range(lag):
    for i in range(len(rgc[l]) - 2, -1, -1):
        rgc[l][i] = lor(rgc[l][i], rgc[l][i + 1])

pathlib.Path('eval/' + folder).mkdir(parents=True, exist_ok=True)

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