
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
We use K-fold. Ten re-sampling of a 10-fold cross-validation were executed producing 100 bootsraps of each datasets. In each iteration, nine-folds of the original data sets were input to each of trhe five regression models 


When implementing K-fold we want the class distribution in the dataset to be preserved in the training and test splits. This means that if, for example, the ratio of <20 years subjects (class0) to >20 years (class1)subject is 1/3. If we set k=4, then the test sets include three data points from class1 and one data point from class 0. Thus, training sets include three data points from class 0 and nine data points from class 1.
This can be done with Stratified Kfold.
We can also extend the binary concept of classo 0 and 1 to multiclass . In particular we are going to divede the dataset in four class.

## Data Harmonization 
As explained by Ferrai et al() The CI of a categorical variable is a score between 0 and 1 representing its impact on a binary classification study.
Can we use this figure of merit for a non classification study?

This score is calculated by replicating the desired classification study (i.e. same task and same classifier) using different degrees of bias in the training set with respect to the hypothetically confounding variable. The CI value summarizes how much the performances of the classifiers change among the various trainings. The CI can be calculated also for continuous variables by binning their values and, in this scenario, it is useful to identify the widest range of values for which the effect of such variables can be ignored.

The paper we are referring to describes the implementation of a classifier. We instead want to predict a value (the age) and we want to examine the confounding effect of a categorical feature ( aquisition site) using the CI index.



To mitigate the effect of the different acquisition sites on the features, we have to harmonize data across sites. Wecan use different models , in particular we want to follow Lombardi et al(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7349402/#B40-brainsci-10-00364).

For the removal of site effects, different harmonization procedures were compared: 
(i) absence of harmonization ;
(ii) Removal of site effects using linear regression without adjusting for biological covariates
(ii) removal of site effects using ComBat with age as biological covariate of interest (age covariate); 
(iii) removal of site effects using ComBat without specifying the age as a biological covariate to be preserved (no age covariate). 
Why do we have to use the age as covariare ?
We have to consider the interaction between the site variable and the age of the subjects.
If some sites only include subjects with age in specific ranges( this is our case) , it is therefore important to ensure that the harmonization of the site effect does not affect the age-related biological variability of the dataset. 


the Combat model we are going to use has been implemented in opython by Fortin et al. (https://www.sciencedirect.com/science/article/abs/pii/S105381191730931X)

## Requirements

To use our Python codes the following packages are needed: numpy, scikit-learn, seaborn, pandas, matplotlib, and scipy.

## How to use
