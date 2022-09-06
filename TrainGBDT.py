import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from utils import read_bin, append_bin, PrepareData


class TrainMLP:
    """

    This class used to train gbdt models

    """

    def __init__(self, time_frame, n_features, datafile_prefix,
                 n_datafiles, label_path):
        self.relevant = None
        self.ans = None
        self.X = None
        self.Y = None
        self.X_test = None
        self.Y_test = None
        self.results = None
        self.time_frame = time_frame
        self.n_features = n_features
        self.n_datafiles = n_datafiles
        self.datafile_prefix = datafile_prefix
        self.label_path = label_path
        self.encoder = OneHotEncoder()

    def merge_data(self):
        """
        Merge all data files together

        :return: Merged data and labels
        """
        relevant = []
        for i in range(1, self.n_datafiles + 1):
            filename = self.datafile_prefix + str(i)
            print("merge data " + str(i))
            data = read_bin(filename)
            for j in range(len(data)):
                relevant.append(data[j].T)
        ans = read_bin(self.label_path)
        self.relevant = relevant
        self.ans = ans

        return relevant, ans

    def spilt_data(self, test_size, random_state):
        """
        Spilt all data into train set and test set

        :param test_size: (float) Proportion of test set
        :param random_state: (int) Random seed
        :return: Train set and test set
        """
        # data = np.array(self.relevant)
        data_train, data_test, ans_train, ans_test = train_test_split(
            np.array(self.relevant),
            self.ans,
            test_size=test_size,
            random_state=random_state)
        self.X, Y = PrepareData(data_train, ans_train, self.time_frame,
                                mode="Normal")
        self.X_test, Y_test = PrepareData(data_test, ans_test,
                                          self.time_frame, mode="Normal")

        self.Y = self.encoder.fit_transform(Y.reshape(len(Y), 1)).toarray()
        self.Y_test = self.encoder.fit_transform(
            Y_test.reshape(len(Y_test), 1)).toarray()

        return self.X, self.Y, self.X_test, self.Y_test

    def train_gbdt(self, replicas, n_estimators, learning_rate, max_depth,
                   subsample, save_models=True, save_plots=True,
                   save_acc=True,
                   save_path='./GBDT_results/bigdata/water/'):
        """

        :param replicas: (int) Number of replications
        :param n_estimators: (int) Number of trees
        :param learning_rate: (float) Set learning rate of gbdt models
        :param max_depth: (int) Set maximum depth of trees
        :param subsample: (float) Set the subsample rate for each train
        :param save_models: (Boolean) Whether to save the models
        :param save_plots: (Boolean) Whether to save the plots
        :param save_acc: (Boolean) Whether to save the accuracy data
        :param save_path: (String) Path to save results
        :return: None
        """
        scores = []
        fi = []
        for i in range(1, replicas + 1):

            clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             max_depth=max_depth,
                                             random_state=random.randrange(
                                                 0, 1000),
                                             subsample=subsample).fit(
                self.X, self.Y)
            if save_models:
                with open(save_path + 'Models/GBDT_models' + str(i),
                          'wb') as f:
                    pickle.dump(clf, f)
                f.close()

            score_train = clf.score(self.X, self.Y)
            score_test = clf.score(self.X_test, self.Y_test)
            gbdt_fi = clf.feature_importances_
            scores.append([score_train, score_test])
            fi.append(gbdt_fi)

            if save_plots:
                plt.plot(gbdt_fi)
                plt.savefig(save_path +
                            'Plots/feature_importances_' + str(i) + '.jpg',
                            bbox_inches='tight')
                plt.close()

        if save_acc:
            file = open(save_path + 'feature_importances_all', "ab+")
            append_bin(file, fi)
            file.close()

            file = open(save_path + 'acc_all', "ab+")
            append_bin(file, scores)
            file.close()



if __name__ == '__main__':
    mytrain = TrainMLP([6750, 7500], 1749, "./CVs_data/CVdata_5A+wat_", 7,
                       "./CVs_data/CVdata_labels")
    mytrain.merge_data()
    mytrain.spilt_data(0.2, 0)
    mytrain.train_gbdt(10, 100, 0.1, 3, 0.8)
