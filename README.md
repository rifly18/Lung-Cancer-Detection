# Pulmonologists-Level lung cancer detection based on standard blood test analyses and smoking status using a machine learning approach: A retrospective model development and validation study. 

This is the github repository corresponding to the article Pulmonologists-Level lung cancer detection based on standard blood test analyses and smoking status using a machine learning approach: A retrospective model development and validation study.

Several steps are needed, in order to setup the experiment and results in this article. The steps are stated below.
## Requirements

The following libraries and the stated version (only for some) should be installed in order for the code to run

- pip install numpy pandas seaborn matplotlib scikit-learn imblearn scipy lightgbm xgboost keras tensorflow shap deslib statsmodels scikit_posthocs orange3==3.30

## Training

In order to train the models in the paper the primary notebook _main.ipynb_ should be run. This will also gather the results of the models.

When running this code, in the _main.ipynb_, specifically in the cross validation section, the user must add the desired model that they wish to train and evaluate as a parameter. 

An example of how to train the LR model:

- CrossValidate('LR', X, y, X_200, y_200) #Cross Validation for Logistic Regression
_Note that both the training data and validation data is inputted, as the results are also returned after training using this line of code_

The different models used in this paper can be found in the _Models.py_ file.

**Keep in mind that it is NOT possible to run this notebook without the two datasets:** 
- CombinedDataset.csv
- ENG_200Samples.csv (the validation data set)

## Evaluation

All results will be produced running the _main.ipynb_ without any additional parameters other than that specified in the training stage.

Evaluation of the models includes:

- TP
- FP
- TN
- FN
- Sensitivity
- Specificity
- TPR
- FPR
- Precision
- F1-score

Additionally different plots (all presented in the paper or the appendix) are produced running the rest of the _main.ipynb_ notebook
