# Lung-Cancer-Detection

This is the github repo corresponding to the article Pulmonologists-Level lung cancer detection based on standard blood
test analyses and smoking status using a machine learning approach: a
retrospective model development and validation study.

Steps to setup the experiments and results
1) pip install numpy pandas seaborn matplotlib scikit-learn imblearn scipy lightgbm xgboost keras tensorflow shap deslib statsmodels scikit_posthocs orange3==3.30
2) main.py is the primary notebook. Keep in mind that it is not possible to run this notebook without the CombinedDataset.csv dataset and the validation data setENG_200Samples.csv files
3) In the cross validation section to find results, type out the specific model that you need as a parameter which is implemented in the Models.py - Options are implemented in this file.
4) Next, all results onwards can be run, without any specified parameters.
