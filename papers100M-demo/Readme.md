# CSP Experiment on Papers100M Dataset  

This repository contains the code and resources for conducting the CSP experiment on the Papers100M dataset.  

## Contents  

- **Notebook:**  
  - `CSP_on_ogbn_papers.ipynb`: Provides an overview of the experiment, including its implementation and evaluation of results.  
  - Includes an SQL query for computation on Spark clusters.  
  - **Note:** The dataset does not need to be downloaded for running this notebook.  

- **Dataset:**  
  - The Papers100M dataset can be downloaded from [Papers100M](https://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip).  
  - After downloading, extract the contents of the ZIP file to a suitable location. This path will be used in the scripts to load the dataset.  

- **Scripts:**  
  1. `create_datasets.py`:  
     - Preprocesses the dataset.  
     - Saves preprocessed tables to disk, which are used in the main script.  

  2. `papers100m_classification.py`:  
     - Runs the main experiment.  
     - Loads the preprocessed tables and applies the CSP algorithm.  
     - Designed to run on smaller devices (16GB RAM + some swap space should suffice).  

## Getting Started  

1. **Download the dataset:** [Papers100M](https://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip).  
2. **Extract the dataset:** Place the extracted files in a suitable directory.  
3. **Run `create_datasets.py`:** Preprocess the dataset and generate required tables.  
4. **Run `papers100m_classification.py`:** Execute the CSP experiment on the preprocessed data.  

