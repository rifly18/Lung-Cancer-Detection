def LoadData():
    import pandas as pd
    
    df=pd.read_csv("Data/CombinedDataset.csv")
    df=df.drop(['lab','lab_year'], axis=1)
    df_200_samples = pd.read_csv("Data/ENG_200Samples.csv")
    
    # Train and validation data
    X=df.drop('LC',axis=1)
    y=df['LC']
    
    # Test data
    X_200 = df_200_samples.drop('LC', axis=1)
    y_200 = df_200_samples['LC']
    
    return X,y,X_200,y_200

def PreprocessData(X_train, y_train, X_val, y_val, X_200, y_200):
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import shuffle
    import pandas as pd
    rus = RandomUnderSampler(random_state=2)
    numeric_medianProcessing = Pipeline(
    [('Imputation', SimpleImputer(strategy='median')),
     ('Scaling', StandardScaler())])
    
    preprocessing = make_column_transformer((numeric_medianProcessing,["ALAT","Amylase","Alkaline phosphatase","Basophils","Bilirubin-total","C-reactive protein","Calcium-total","Eosinophils","INR","Creatinine",
                                                                       "LDH","Leucocytes","Lymphocytes","Monocytes","Neutrophils","Platelets","Age","Albumin","Hemoglobin","Sodium",'Potassium','Gender','Smoking status']))


     # Fit pipeline
    X_train = preprocessing.fit_transform(X_train)
    X_val = preprocessing.transform(X_val)
    
    if isinstance(X_200, pd.DataFrame):
        X_200 = preprocessing.transform(X_200)
        
    
    # Undersample
    X_train,y_train = rus.fit_resample(X_train,y_train)
    
    #Shuffle data
    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val = shuffle(X_val, y_val)
    X_200, y_200 = shuffle(X_200, y_200)
    
    return X_train, y_train, X_val, y_val, X_200, y_200