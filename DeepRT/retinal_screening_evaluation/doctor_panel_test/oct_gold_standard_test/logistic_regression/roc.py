import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
print(__doc__)
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from load_answers import load_answers
le = LabelEncoder()

gold_standard = pd.read_csv("./gold_standard_answers/gs_variations_a_e_n.csv")[["id", "A/N/E_400"]]
gold_standard.id = gold_standard.id.str.replace(".png","")

# load all paths
fundus_answ_paths = glob.glob("./*/*answering_sheet_fundus_prediction.csv")

data = load_answers(fundus_answ_paths, gold_standard)

# Run classifier with cross-validation and plot ROC curves


classifier= OneVsRestClassifier(LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial'))

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# set data variables
X = np.array(data[["answer","record_id","doctor"]]).astype(int)
y = np.array(data[["y"]]).astype(int)

# Binarize the output
y_bin = label_binarize( y, classes = [0, 1, 2] )
n_classes = y.shape[1]

cv = StratifiedKFold(n_splits=4)
fpr = [[],[],[],[]]
tpr = [[],[],[],[]]
roc_auc = [[],[],[],[]]
iter_ = 0
for train_index, test_index in cv.split(X, y):
    # shuffle and split training and test sets
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y_bin[train_index]
    y_test = y_bin[test_index]

    # Learn to predict each class against the other
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        #fpr[i], tpr[i], _ =+ roc_curve(y_test[:, i], y_score[:, i])
        fpr[i] = roc_curve( y_test[:, i], y_score[:, i] )[0].tolist()
        tpr[i] = roc_curve( y_test[:, i], y_score[:, i] )[1].tolist()

        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
from itertools import cycle

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes


# Plot all ROC curves
plt.figure()

classes = ["atrophy", "edema", "normal"]
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()