import pandas as pd
import matplotlib.pyplot as plt

def tsne(X,y, k_fold, p_time):
    """

    :param x:
    :param y:
    :param savep_path:
    :return:
    """


    from sklearn.manifold import TSNE
    from cycler import cycler

    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化


    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])

    #plt.show()

    plt.savefig('{}-fold_tSNE.png'.format(k_fold))
    plt.close()

    save_path_excel = '{}-fold_tSNE.xls'.format(k_fold)
    pd.set_option('display.max_columns', None)
    each_fold = pd.DataFrame(index=[str(i) for i in range(len(X_norm))])
    each_fold["tsne_f_1"] = X_norm[:, 0]
    each_fold["tsne_f_2"] = X_norm[:, 1]
    each_fold["label"] = y
    each_fold.to_excel(save_path_excel)

