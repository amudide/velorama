import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import metrics
import os
import glob

def lor(x, y):
    return [a or b for a, b in zip(x, y)]

folder = 'De-noised_100G_6T_300cPerT_dynamics_7_DS6-True-5000-[100]-5-H-0-0.0001'

lams = np.logspace(-3.0, 3.0, num=39).tolist()

mat1 = pd.read_csv('img/' + folder + "/1.0/preds.csv", index_col = 0)
gt = mat1['y_true'].tolist()

mats = []

for lam in lams:
    mat = pd.read_csv('img/' + folder + "/" + str(lam) + "/preds.csv", index_col = 0)
    mat = mat['y_probas'].tolist()
    
    for i, val in enumerate(mat):
        mat[i] = (val > 1e-8)
    
    mats.append(mat)
    
rmats = []

for i, mat in enumerate(mats):
    pzero = sum(mat) / len(mat)
    if pzero >= 0.05 and pzero <= 0.99:
        rmats.append(mat)
        
for i in range(len(rmats) - 2, -1, -1):
    rmats[i] = lor(rmats[i], rmats[i + 1])
    
portions = sum(np.array(rmat) for rmat in rmats)

portions = portions / len(rmats)

fpr, tpr, threshold = metrics.roc_curve(gt, portions)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig('eval/' + folder + '.png', bbox_inches='tight')
plt.show()

