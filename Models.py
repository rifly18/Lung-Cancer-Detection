def Model(modelName, X_train, y_train):
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    import lightgbm as lgb
    import xgboost as xgb
    from deslib.dcs.ola import OLA

    if(modelName=='LR'):
        clf = LogisticRegression(C = 0.30000000000000004, penalty= 'l2', solver= 'lbfgs',random_state=2).fit(X_train,y_train)
        
    elif(modelName=='SVM'):
        clf = SVC(C=49.1,kernel='linear',probability=True,random_state =2).fit(X_train,y_train)
        
    elif(modelName=='XGB'):
        clf = xgb.XGBClassifier(objective='binary:logistic',seed = 42,eval_metric = 'auc',eta=0.02,gamma=0,max_depth=3,min_child_weight = 8, n_estimators = 765,random_state =2).fit(X_train,y_train)
        
    elif(modelName=='LGBM'):
        clf = lgb.LGBMClassifier(learning_rate = 0.1,max_depth =1,min_data_in_leaf = 17,min_gain_to_split = 0,n_estimators = 210, num_leaves = 550,random_state=2).fit(X_train,y_train)
        
    elif(modelName=='DES'):
        classifiers = [
            LGBMClassifier(learning_rate = 0.1,max_depth =1,min_data_in_leaf = 17,min_gain_to_split = 0,n_estimators = 210, num_leaves = 550,random_state =2),
            XGBClassifier(objective='binary:logistic',seed = 42,eval_metric = 'auc',eta=0.02,gamma=0,max_depth=3,min_child_weight = 8, n_estimators = 765,random_state =2),
            LogisticRegression(C = 0.30000000000000004, penalty= 'l2', solver= 'lbfgs',random_state =2),
            SVC(C=49.1,kernel='linear',probability=True,random_state =2)]
        
        for c in classifiers:
            c.fit(X_train, y_train)

        clf = OLA(pool_classifiers=classifiers).fit(X_train, y_train)
        
    else:
        print('Provide name of model as first input parameter, nothing is returned')
        
    return clf