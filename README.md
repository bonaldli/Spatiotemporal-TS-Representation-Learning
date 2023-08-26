# Spatiotemporal Self-supervised Bootstrapping
This repository contains the official implementation for the paper Correlated Time Series Self-Supervised Representation Learning via Spatiotemporal Bootstrapping (CASE-23).

## Requieremnets
torch==1.13.1 \
numpy==1.22.3 \
scikit_learn==1.1.1 \
tensorboard==2.9.0 

The dependencies can be installed by:

`pip install -r requirements.txt`

## Data
* **METR-LA**. This traffic dataset contains traffic flow records collected from loop detectors in the highway of Los Angeles County. There are four months
of data ranging from Mar 1st, 2012 to Jun 30th, 2012
collected at 207 sensors for the experiment.

* **PEMS-BAY**. This traffic dataset is collected by California Transportation Agencies (CalTrans) Performance Measurement System (PeMS). There are three months of data ranging from Jan 1st, 2017, to May 31st, 2017, collected at 325 sensors for the experiment.

* **MTR**. This is a real-world passenger inflow dataset, Hong Kong Metro, including data from 90 metro stations in the first three months of 2017 and 93 stations in June 2021. 3 new stations were opened between these years. The raw data is aggregated into 5 minutes, and we have 247 time steps in one day due to the operation time of the metro system.

**Remark**: METR-LA and Pems-Bay can be found in the folder datasets. For privacy reason, MTR dataset will not be published, but the code of cold-start task related to this dataset can be found in the python file cold_start.py.

## Usage
To train the representation:
`python train.py --name <dataset> --epoch <epoch> --lamda <spb hyperparameter> --device <GPU to use>`

Remark: The trained encoders are in the folder **model**.

To evaluate on downstream tasks and conduct ablation study:
`python forecasting.py --name <dataset> --lamda <spb hyperparameter> --missing_rate <missing rate of test data> --device <GPU to use>`


