# CDK2_CV_Selection

<a href="https://github.com/allegroai/clearml"><img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white "></a>

This project is an individual project for the PHAS0077 module at University College London.

This project applied MLP and GBDT machine learning methods to learn and predict simulation data for the 3sw4 structure of CDK2 and filtered key collective variables by MLTSA.



## Files Structure:

- [**Project_files:**](./Project_files) Initial structure and trajectory files (Only a test trajectory file is uploaded).

- **[CVs_data:](./CVs_data)** The name of the collective variables for each dataset and the label of all the datasets.

- [**GBDT_results:**](./GBDT_results) All the results of GBDT models (Including 2 big datasets and 2 small datasets).
  - **[bigdata:](./GBDT_results/bigdata)** Containing 2 big datasets' results (One dataset contains water residues another one not).
  - **[smalldata:](./GBDT_results/smalldata)** Containing 2 small datasets' results (One dataset contains water residues another one not).

- **[MLP_results:](./MLP_results)** All the results of MLP models (Including 2 big datasets and 2 small datasets). 
  - **[bigdata:](./MLP_results/bigdata)** Containing 2 big datasets' results (One dataset contains water residues another one not).
  - **[smalldata:](./MLP_results/smalldata)** Containing 2 small datasets' results (One dataset contains water residues another one not).

- **[Notebook_Results:](./Notebook_Results)**  Summary of the final CV selecting results for the four datasets

  **[CV_from_MD.py:](./CV_from_MD.py)** It can be used to analyze the molecular dynamics generated on the simulations as dcd files.

  **[GenerateData.py:](./GenerateData.py)** The class can generate datasets and labels from simulation files in preparation for machine learning.

  **[utils.py:](./utils.py)** A number of utility methods, including reading and writing files, preparing data, etc.

  **[MLTSA_sk.py:](./MLTSA_sk.py)** Apply the Machine Learning Transition State Analysis to SK_learn model.

  **[MLTSA_tf.py:](./MLTSA_tf.py)** Apply the Machine Learning Transition State Analysis to Tensorflow model.

  **[TF_2_MLP.py:](./TF_2_MLP.py)** Including some MLP architectures' creating methods, can be used to build MLP models.

  **[TrainGBDT.py:](./TrainGBDT.py)** This class can be used to train GBDT models and save results to specific path.

  **[TrainMLP.py:](TrainMLP.py)** This class can be used to train MLP models and save results to specific path.


## Usage:

To use this code, you need to ensure that your device has a python environment and that some packages are installed. The versions of python and the packages used in this project are as follows (if you are using a different version, please refer to the relevant documentation for support)： 

**python:** 3.9  
**mdtraj:** 1.9.7  
**tensorflow:** 2.8.0  
**scikit-learn:** 1.0.2  
**numpy:** 1.22.3  
**matplotlib:** 3.5.1  

