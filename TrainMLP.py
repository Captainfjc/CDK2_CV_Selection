import TF_2_MLP
import numpy as np
import pickle
import matplotlib.pyplot as plt
from MLTSA_tf import MLTSA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from utils import read_bin, append_bin, PrepareData


class TrainMLP:
    """

    This class used to train mlp models

    """

    def __init__(self, time_frame, n_features, datafile_prefix, n_datafiles, label_path):
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

    def train_mlp(self, replicas, epochs, type, save_models=True,save_plots=True,save_acc=True,save_path='./MLP_results/bigdata/water/'):
        """
        Train mlp model

        :param replicas: (int) Number of replications
        :param epochs: (int) Maximum epoches for the models to train
        :param type: (String) Model type, choose the model from 'TF_2_MLP.py'
        :param save_models: (Boolean) Whether to save the models
        :param save_plots: (Boolean) Whether to save the plots
        :param save_acc: (Boolean) Whether to save the accuracy data
        :param save_path:  (String) Path to save results
        :return: None
        """
        mltsa = []
        train_acc_list = []
        test_acc_list = []
        val_acc_list = []
        for i in range(1, replicas + 1):
            print("replicas: " + str(i))
            n_steps = self.time_frame[1] - self.time_frame[0]
            MLP = TF_2_MLP.build_MLP(n_steps, self.n_features, n_labels=2,
                                     type=type).model
            train_log = MLP.fit(self.X, self.Y, epochs=epochs,
                                batch_size=n_steps, verbose=1,
                                validation_split=0.2,
                                callbacks=[
                                    EarlyStopping(monitor='val_accuracy',
                                                  min_delta=1e-4,
                                                  restore_best_weights=True,
                                                  patience=50,
                                                  mode='max')])
            test_acc = MLP.evaluate(self.X_test, self.Y_test, verbose=1)
            print("We achieved", test_acc[1] * 100,
                  "% accuracy on Testing and ",
                  max(train_log.history["accuracy"]) * 100,
                  "% accuracy on Training")

            # Calculate accuracy and loss
            loss = train_log.history["loss"]
            val_loss = train_log.history["val_loss"]
            acc_train = train_log.history["accuracy"]
            acc_val = train_log.history["val_accuracy"]

            # Calculate accuracy drop
            a_drop = MLTSA(np.array(self.relevant)[:, :,
                           self.time_frame[0]:self.time_frame[1]],
                           self.ans, MLP, self.encoder)

            # plot accuracy and loss
            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            ax[0].plot(np.array(acc_train) * 100, color="r",
                       label="Training accuracy")
            ax[0].plot(np.array(acc_val) * 100, color="g",
                       label="Validation accuracy")
            ax[0].legend()
            ax[0].set_xlabel("Epochs")
            ax[0].set_ylabel("Accuracy")

            ax[1].plot(loss, label="training loss", color="r")
            ax[1].plot(val_loss, label="validation loss", color="g")
            ax[1].legend()
            ax[1].set_xlabel("Epochs")
            ax[1].set_ylabel("loss")
            ax[1].set_ylim(0, 1)

            plt.plot(np.mean(a_drop, axis=1))
            plt.xlabel("Features")
            plt.ylabel("Accuracy (%)")
            if save_plots:
                plt.savefig(save_path + 'Plots/MLTSA_' + str(i) + '.jpg',
                            bbox_inches='tight')
            plt.close()

            mltsa.append(a_drop)
            train_acc_list.append(max(train_log.history["accuracy"]))
            val_acc_list.append(max(train_log.history["val_accuracy"]))
            test_acc_list.append(test_acc)

            if save_models:
                with open(save_path + 'Models/MLP_models'+str(i),
                          'wb') as f:
                    pickle.dump(MLP, f)
                f.close()

        if save_acc:
            file = open(save_path + 'a_drop_all', "ab+")
            append_bin(file, mltsa)
            file.close()

            file = open(save_path + 'acc_all', "ab+")
            append_bin(file, train_acc_list)
            append_bin(file, val_acc_list)
            append_bin(file, test_acc_list)
            file.close()


if __name__ == '__main__':
    mytrain = TrainMLP([6750, 7500], 1749, "./CVs_data/CVdata_5A+wat_", 7, "./CVs_data/CVdata_labels")
    mytrain.merge_data()
    mytrain.spilt_data(0.2, 0)
    mytrain.train_mlp(50, 1000, 'simple')
