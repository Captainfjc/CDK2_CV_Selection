import pickle
import numpy as np


def read_bin(filename):
    """
    Read binary files

    :param filename: (str) Path to the binary file to read
    :return: (list) The data in the file
    """

    data = []
    with open(filename, "rb") as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    # #data = pickle.load(file)
    return data


def create_bin(filename, data):
    """
    Create a binary file

    :param filename: (str) Path and name of the binary file to create
    :param data: The data to put in the file
    :return: The file stream
    """

    file = open(filename, "wb+")
    pickle.dump(data, file)
    # file.close()
    return file


def append_bin(file, data):
    """
    Add things to an existing binary file

    :param file: IO stream of the file
    :param data: The data to add
    :return: The data added
    """
    # file = open(filename, "ab")
    pickle.dump(data, file)
    # file.close()
    return data


def PrepareData(data, ans, time_frame, mode="Normal"):
    """

    Small wrapper that prepares the data inputted from the dataset object as the correct format to use on the ML
    approach.

    :param data: (list) Data simulation generated from the dataset.
    :param ans: (list) Labels of the outcome from the simulations
    :param time_frame: (list) [start_frame_range, end_frame_range] Values for the range of frames/steps to keep for
        the final data, this allows to select a particular amount from the trajectories.
    :param mode: (str) Wether to use the real value of the relevant potential as a last feature or not. "Normal"
        means using it.
    :return: (list) List containing the data as (X, Y) being X the simulation data as the mixed trajectories and
        Y as the labelled outcomes for each frame.

    """
    X = data[:, :, time_frame[0]: time_frame[1]]
    X = np.concatenate(X, axis=1).T
    Y = np.ones(len(X)).astype(str)
    print(len(X))
    for n, answer in enumerate(ans):
        frames = time_frame[1] - time_frame[0]
        tmp_ans = np.ones(frames).astype(str)
        tmp_ans[:] = answer
        Y[n * frames:n * frames + frames] = tmp_ans

    if mode == "Normal":
        pass
    elif mode == "Rigged":
        X = X[:, :-1]
    print("Prepare finish")
    return X, Y
