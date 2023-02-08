import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os

# The first kind of confusion matrix
def plot_confusion_matrix_1(pre_lab, Y_test, num_p, logger):
    acc = accuracy_score(Y_test, pre_lab)
    logger.info('The prediction accuracy on this fold is: {:.4f}\n'.format(acc))

    logger.info('Ground of Truth')
    logger.info(Y_test)
    logger.info('Predicted label')
    logger.info(pre_lab)


    class_label = ['ANGER', 'CONTEMPT','DISGUST', 'FEAR', 'HAPPY','NEUTRAL', 'SADNESS', 'SURPRISE']
    conf_mat = confusion_matrix(Y_test, pre_lab)
    df_cm = pd.DataFrame(
        conf_mat,
        index=class_label,
        columns=class_label
    )

    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=25, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Backbone_d--{} time 10-fold-cross-validation cm_1.png'.format(num_p))
    plt.show()

    return

# The first kind of confusion matrix
def plot_confusion_matrix_2(pre_lab, Y_test):


    labels_name = ['Happiness','Sadness','Neutral','Anger','Surprise', 'Disgust','Fear' ]
    cm = confusion_matrix(Y_test, pre_lab)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    ind_array = np.arange(len(labels_name))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            if x_val == y_val:
                plt.text(x_val, y_val, "%.2f" % (c,), color='white', fontsize=10, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%.2f" % (c,), color='black', fontsize=10, va='center', ha='center')
    confusion_matrix_2(cm_normalized, labels_name, 'Normalized Confusion Matrix')
    # os.remove("./dfew.png")
    plt.savefig("./dfew.png")
    plt.close()
    # plt.show()

    return


def confusion_matrix_2(cm, labels_name, title):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.PuRd)
    # plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=25)
    plt.yticks(num_local, labels_name)
    # plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return














