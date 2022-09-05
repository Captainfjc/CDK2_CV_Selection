import CV_from_MD as cvs
import mdtraj as md
from pathlib import Path
from utils import append_bin


class Dataset:
    """

    This class is used to generate datasets and labels from simulation files in preparation for machine learning

    """

    def __init__(self, top_path):
        self.top_path = top_path
        self.CV_system = cvs.CVs(top_path)
        self.dcd_path_list = []
        self.analyzer = cvs.MDs()

    def select_cvs(self, dcd_path, distance):
        """
        Select collective variables from the specified structure file and trajectory file

        :param dcd_path: (str) Trajectory file path of simulation
        :param distance: (int) Distance to define close atoms
        :return: (list) The indices of atoms of selected CVs
        """

        TS = md.load_frame(dcd_path, 0, self.CV_system.topology_file)
        relevant_atoms = list(
            self.CV_system.top.select("not name H and resname LIG"))
        close_atoms = list(
            md.compute_neighbors(TS, distance, relevant_atoms)[0])
        all_combinations = [[i, j] for i in relevant_atoms for j in
                            close_atoms]
        new_CV = []
        for pair in all_combinations:
            if pair[0] in relevant_atoms and pair[1] in relevant_atoms:
                pass
            else:
                new_CV.append(pair)
        self.CV_system.define_variables("custom_CVs", CV_indices=new_CV)
        return new_CV

    def generate_data(self, n_dcd, n_sim_per_file, dcd_path_prefix,
                      dcd_path_suffix, datafile_name):
        """
        Generate a dataset at the specified path, the dataset will be split
        into multiple files and the caller can specify how many simulations
        each file contains

        :param n_dcd: Number of dcd files
        :type n_dcd: int
        :param n_sim_per_file: Number of simulations contained in one file
        :type n_sim_per_file: int
        :param dcd_path_prefix: dcd files path prefix
        :type dcd_path_prefix: str
        :param dcd_path_suffix: dcd files path suffix
        :type dcd_path_suffix: str
        :param datafile_name: Save path for the generated dataset
        :type datafile_name: str
        :return:
        """
        for i in range(n_dcd):
            self.dcd_path_list.append(
                [dcd_path_prefix + str(i + 1) + dcd_path_suffix])
        count_file = 1
        count_sim = 1
        file = open(datafile_name + str(count_file), "wb+")
        for i in range(len(self.dcd_path_list)):
            if count_sim == 1:
                bin_filename = datafile_name + str(
                    count_file)  # File name to save data
                file = open(bin_filename, "wb+")
            my_file = Path(self.dcd_path_list[i][0])
            if my_file.exists():
                dcd_path = self.dcd_path_list[i]
                CVs_from_bubble = self.analyzer.calculate_CVs(
                    self.CV_system, dcd_path)
                append_bin(file, CVs_from_bubble[0])
                count_sim += 1
                if count_sim == n_sim_per_file + 1:
                    file.close()
                    count_sim = 1
                    count_file += 1
            else:
                count_sim += 1
        file.close()

    def generate_labels(self, label_filename, selection_strings_to_label):
        """
        Generate labels for the generated data sets and store them into a binary file

        :param label_filename:
        :type label_filename: str
        :param selection_strings_to_label: String selection using mdtraj's atom selection reference syntax.
        :type selection_strings_to_label: list
        :return:
        """
        label_list = []
        for i in range(len(self.dcd_path_list)):
            my_file = Path(self.dcd_path_list[i][0])
            if my_file.exists():
                dcd_path = self.dcd_path_list[i]
                labels = self.analyzer.label_simulations(self.top_path,
                                                         dcd_path,
                                                         selection_strings_to_label,
                                                         9, 9,
                                                         plotting=False,
                                                         show_plots=False,
                                                         save_labels=False,
                                                         save_path="test/test_")
                label_list.append(labels[0])
        file = open(label_filename, "ab+")
        append_bin(file, label_list)
        file.close()


if __name__ == '__main__':
    CVData = Dataset('./Project_files/new_top_cropped.pdb')
    CVData.generate_data(160, 25, "../../hankang_backups/MDdata/3sw4/set1/TS_", "_crop.dcd", "./CVs_data/CVdata_")
    CVData.generate_labels('./CVs_data/CVdata_labels', [["residue 83 and name O",
                                                        "resname LIG and name N9"],
                                                        ["residue 83 and name N",
                                                        "resname LIG and name N7"]])



















