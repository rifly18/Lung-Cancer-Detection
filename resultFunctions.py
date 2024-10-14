def CMResults(X_data, y_data, clf):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_data, clf.predict(X_data)).ravel()

def Metrics(X_data, y_data, clf):
    from sklearn.metrics import precision_recall_fscore_support
    acc = clf.score(X_data, y_data)
    ppv = precision_recall_fscore_support(y_data, clf.predict(X_data))[0][1]
    se = precision_recall_fscore_support(y_data, clf.predict(X_data))[1][1]
    f1 = precision_recall_fscore_support(y_data, clf.predict(X_data))[2][1]
    return acc, ppv, se, f1

def MetricsToDf(dfName, X_data, y_data, clf):
    import pandas as pd
    acc, ppv, se, f1 = Metrics(X_data, y_data, clf)
    TN, FP, FN, TP = CMResults(X_data,y_data, clf)
    
    dfName.loc[len(dfName)] = {'accuracy': acc,
                               'precision': ppv,
                               'recall': se,
                               'f1': f1,
                               'specificity': TN/(TN+FP),
                               'TN': TN,
                               'FP': FP,
                               'FN': FN,
                               'TP': TP}
    return dfName


def ROC(dfName, X_data, y_data, clf):
    from sklearn import metrics
    from sklearn.metrics import roc_auc_score,roc_curve
    dfName.loc[len(dfName)] = {'FPR': metrics.roc_curve(y_data,  clf.predict_proba(X_data)[:,1])[0],
                               'TPR': metrics.roc_curve(y_data,  clf.predict_proba(X_data)[:,1])[1],
                               'thresholds': metrics.roc_curve(y_data, clf.predict_proba(X_data)[:,1])[2],
                               'auc': roc_auc_score(y_data, clf.predict_proba(X_data)[:,1])}
    return dfName

def Predictions(dfName, X_data, y_data, clf):
    import pandas as pd
    dfName.loc[len(dfName)] = {'prediction': clf.predict(X_data),
                               'probability': clf.predict_proba(X_data)[:,1],
                               'actual': y_data}    
    return dfName

def PR(dfName, X_data, y_data, clf):
    import pandas as pd
    from sklearn.metrics import precision_recall_curve,average_precision_score
    dfName.loc[len(dfName)] = {'precision': precision_recall_curve(y_data, clf.predict_proba(X_data)[:,1])[0],
                               'recall': precision_recall_curve(y_data, clf.predict_proba(X_data)[:,1])[1],
                               'thresholds': precision_recall_curve(y_data, clf.predict_proba(X_data)[:,1])[2],
                               'auc': average_precision_score(y_data, clf.predict_proba(X_data)[:, 1])}
    return dfName
    


def Evaluate(clf, X_train, y_train, X_val, y_val, X_200, y_200, dfTrainResults, dfValResults, df200Results, dfValROC, df200ROC, dfValPR, df200PR, dfValPreds, df200Preds, dfValCalib, df200Calib):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    import pandas as pd
    if dfTrainResults.empty:
        dfTrainResults = pd.DataFrame(columns=['accuracy'])
    
    if dfValResults.empty:
        dfValResults = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'specificity', 'TN', 'FP', 'FN', 'TP'])
        
    if df200Results.empty:
        df200Results = pd.DataFrame(columns=['accuracy', 'recall', 'precision', 'f1', 'specificity', 'TN', 'FP', 'FN', 'TP'])
        
    if dfValROC.empty:
        dfValROC = pd.DataFrame(columns=['FPR', 'TPR', 'thresholds', 'auc'])
    
    if df200ROC.empty:
        df200ROC = pd.DataFrame(columns=['FPR', 'TPR', 'thresholds', 'auc'])
        
    if dfValPR.empty:
        dfValPR = pd.DataFrame(columns=['precision', 'recall', 'thresholds', 'auc'])
        
    if df200PR.empty:
        df200PR = pd.DataFrame(columns=['precision', 'recall', 'thresholds', 'auc'])
    
    if dfValPreds.empty:
        dfValPreds = pd.DataFrame(columns=['prediction', 'probability', 'actual'])
        
    if df200Preds.empty:
        df200Preds = pd.DataFrame(columns=['prediction', 'probability', 'actual'])
    
    if dfValCalib.empty:
        dfValCalib = pd.DataFrame(columns=['mean_predicted_probability', 'fraction_of_positives', 'brier_score'])

    if df200Calib.empty:
        df200Calib = pd.DataFrame(columns=['mean_predicted_probability', 'fraction_of_positives', 'brier_score'])

    # Append training results
    dfTrainResults = pd.concat([dfTrainResults, pd.DataFrame({'accuracy': [clf.score(X_train, y_train)]})], ignore_index=True)
    
    # Append validation results
    dfValResults = MetricsToDf(dfValResults, X_val, y_val, clf)
    df200Results = MetricsToDf(df200Results, X_200, y_200, clf)
    
    # ROC curves
    dfValROC = ROC(dfValROC, X_val, y_val, clf)
    df200ROC = ROC(df200ROC, X_200, y_200, clf)
    
    # Precision-Recall curves
    dfValPR = PR(dfValPR, X_val, y_val, clf)
    df200PR = PR(df200PR, X_200, y_200, clf)
    
    # Predictions
    dfValPreds = Predictions(dfValPreds, X_val, y_val, clf)
    df200Preds = Predictions(df200Preds, X_200, y_200, clf)
    
    # Calibration Curve and Brier Score for Validation Set
    test_proba_val = clf.predict_proba(X_val)[:, 1]
    brier_val = brier_score_loss(y_val, test_proba_val)
    fraction_of_positives_val, mean_predicted_value_val = calibration_curve(y_val, test_proba_val, n_bins=10)

    dfValCalib = pd.concat([dfValCalib, pd.DataFrame({
        'mean_predicted_probability': [mean_predicted_value_val],
        'fraction_of_positives': [fraction_of_positives_val],
        'brier_score': [brier_val]
    })], ignore_index=True)
    
    # Calibration Curve and Brier Score for X_200 Set
    test_proba_200 = clf.predict_proba(X_200)[:, 1]
    brier_200 = brier_score_loss(y_200, test_proba_200)
    fraction_of_positives_200, mean_predicted_value_200 = calibration_curve(y_200, test_proba_200, n_bins=10)
    
    df200Calib = pd.concat([df200Calib, pd.DataFrame({
        'mean_predicted_probability': [mean_predicted_value_200],
        'fraction_of_positives': [fraction_of_positives_200],
        'brier_score': [brier_200]
    })], ignore_index=True)

    return (dfTrainResults, dfValResults, df200Results, dfValROC, df200ROC, dfValPR, df200PR, dfValPreds, df200Preds, dfValCalib, df200Calib)




def CrossValidate(modelName, X, y, X_200, y_200):
    from Models import Model
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    dfTrainResults = pd.DataFrame()
    dfValResults = pd.DataFrame()
    df200Results = pd.DataFrame()
    dfValROC = pd.DataFrame()
    df200ROC = pd.DataFrame()
    dfValPR = pd.DataFrame()
    df200PR = pd.DataFrame()
    dfValPreds = pd.DataFrame()
    df200Preds = pd.DataFrame()
    bestAUC = 0
    dfValCalib = pd.DataFrame()
    df200Calib = pd.DataFrame()
    
    skf = StratifiedKFold(n_splits=5)
    
    # Iterate through the folds and split into train and test sets
    for train_index, test_index in skf.split(X, y):
        from dataHandling import PreprocessData
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        X_train, y_train, X_val, y_val, X_200, y_200,X_cal,y_cal = PreprocessData(X_train, y_train, X_val, y_val, X_200, y_200,modelName)
        
        clf = Model(modelName, X_train, y_train,X_cal,y_cal)
        
        dfTrainResults, dfValResults, df200Results, dfValROC, df200ROC, dfValPR, df200PR, dfValPreds, df200Preds, dfValCalib, df200Calib = Evaluate(clf, X_train, y_train, X_val, y_val, X_200, y_200, dfTrainResults, dfValResults, df200Results, dfValROC, df200ROC, dfValPR, df200PR, dfValPreds, df200Preds, dfValCalib, df200Calib)
        
        # Best model
        auc= roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
        if  auc>bestAUC:
            bestAUC = auc
            bestModel = clf
        
    return dfTrainResults, dfValResults, df200Results, dfValROC, df200ROC, dfValPR, df200PR, dfValPreds, df200Preds, bestModel, X_val, y_val, dfValCalib, df200Calib