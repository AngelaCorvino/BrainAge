
[![AngelaCorvino](https://circleci.com/gh/AngelaCorvino/BrainAge.svg?style=shield)](https://app.circleci.com/pipelines/github/AngelaCorvino/BrainAge?branch=main&filter=all)

[![Documentation Status](https://readthedocs.org/projects/brainage/badge/?version=latest)](https://brainage.readthedocs.io/en/latest/?badge=latest)


# BrainAge

This repository belongs to Angela Corvino and Agata Minnocci. It contains our project exam for the course of Computing Methods for Experimental Physics and Data Analysis.

The aim of our project is to implement an algorithm that has the ability to predict brain age analysing features extracted from brain MRIs of a cohort of subjects with Autism Spectrum Disorder and control. The algorithm will also compare different regression models and evaluate their performance on this task.

## Data

The features are contained in a .csv file in the BrainAge/data folder.
This dataset contains 419 brain morphological features (volumes, thickness, area, etc.) of brain parcels and global measure, derived for 915 male subjects of the ABIDE I dataset (http://fcon_1000.projects.nitrc.org/indi/abide/). ABIDE stands for Autism Brain Imaging Data Exchange.
The features were computed by means of the FreeSurfer segmentation software. A subsample of the large amount of features generated by Freesurfer for the ABIDE I data cohort is analyzed.

## Feautures Selection

Several feature selection techniques can be applied to remove irrelevant, noisy, and redundant features, avoiding overfitting and improving prediction performance, reducing the computational complexity of the learning algorithm, and proving a deeper insight into the data, which highlights which of the features are most informative for age prediction [[An Introduction to Variable and Feature Selection](https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf?ref=driverlayer.com/web)]

## Data Agumentation
we use K-fold.
When implementing K-fold we want the class distribution in the dataset to be preserved in the training and test splits. This means that if, for example, the ratio of <20 years subjects (class0) to >20 years (class1)subject is 1/3. If we set k=4, then the test sets include three data points from class1 and one data point from class 0. Thus, training sets include three data points from class 0 and nine data points from class 1.
This can be done with Stratified Kfold.
We can also extend the. inary concept of classo 0 and 1 to multiclass .

## Requirements

To use our Python codes the following packages are needed: numpy, scikit-learn, seaborn, pandas, matplotlib, and scipy.

## How to use
