def AppendMetricsResults(dfResults, specialistName):
    import pandas as pd
    dfResults[specialistName].loc['Sensitivity'] = dfResults[specialistName].loc['TP']/(dfResults[specialistName].loc['TP']+dfResults[specialistName].loc['FN'])
    dfResults[specialistName].loc['Specificity'] = dfResults[specialistName].loc['TN']/(dfResults[specialistName].loc['TN']+dfResults[specialistName].loc['FP'])
    dfResults[specialistName].loc['TPR'] = dfResults[specialistName].loc['TP']/(dfResults[specialistName].loc['TP']+dfResults[specialistName].loc['FN'])
    dfResults[specialistName].loc['FPR'] = dfResults[specialistName].loc['FP']/(dfResults[specialistName].loc['TN']+dfResults[specialistName].loc['FP'])
    dfResults[specialistName].loc['Precision'] = dfResults[specialistName].loc['TP']/(dfResults[specialistName].loc['TP']+dfResults[specialistName].loc['FP'])
    dfResults[specialistName].loc['F1'] = (2*dfResults[specialistName].loc['Precision']*dfResults[specialistName].loc['Sensitivity'])/(dfResults[specialistName].loc['Precision']+dfResults[specialistName].loc['Sensitivity'])
    
    return dfResults

def AppendResults(dfCombined, dfResults, specialistName):
    import pandas as pd
    TP=0
    FP=0
    TN=0
    FN=0

    for each in range(len(dfCombined)):
        if dfCombined[specialistName][each]==1 and dfCombined['LC'][each]==1:
            TP=TP+1

        elif dfCombined[specialistName][each]==1 and dfCombined['LC'][each]==0:
            FP=FP+1

        elif dfCombined[specialistName][each]==0 and dfCombined['LC'][each]==0:
            TN=TN+1

        elif dfCombined[specialistName][each]==0 and dfCombined['LC'][each]==1:
            FN=FN+1

    dfResults[specialistName].loc['TP']=TP
    dfResults[specialistName].loc['FP']=FP
    dfResults[specialistName].loc['TN']=TN
    dfResults[specialistName].loc['FN']=FN 
    
    dfResults = AppendMetricsResults(dfResults, specialistName)
    
    return dfResults

def CalculateResults_Majority(dfCombined, dfResults, specialistName):
    import pandas as pd
    import numpy as np
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for each in range(len(dfCombined)):
        specialists = ['Specialist 1', 'Specialist 2', 'Specialist 3', 'Specialist 4', 'Specialist 5']
        lc_value = dfCombined['LC'][each]
        lc_count = sum(dfCombined[specialist][each] for specialist in specialists if not np.isnan(dfCombined[specialist][each]))

        if lc_count >= 3:
            if lc_value == 1:
                TP += 1
            else:
                FP += 1
        else:
            if lc_value == 1:
                FN += 1
            else:
                TN += 1
                
    dfResults[specialistName].loc['TP']=TP
    dfResults[specialistName].loc['FP']=FP
    dfResults[specialistName].loc['TN']=TN
    dfResults[specialistName].loc['FN']=FN 
    
    dfResults = AppendMetricsResults(dfResults, specialistName)

    return dfResults

def SpecialistsResults():
    import pandas as pd
    
    dfSpecialists = pd.read_csv("Data/Answers.csv")
    dfActual = pd.read_csv("Data/200SamplesSmoking.csv")
    
    dfActual = dfActual.rename(columns={"Unnamed: 0":"Patient"})
    dfActual = dfActual[["Patient","LC"]]
    
    # Remove patients that was dropped
    dfActual = dfActual.loc[dfActual["Patient"] != 1622]
    dfActual = dfActual.loc[dfActual["Patient"] != 2640]
    dfActual = dfActual.loc[dfActual["Patient"] != 1215]
    
    # Combine dataframes
    dfCombined = dfSpecialists.join(dfActual.set_index('Patient'), on='Pt. Nr.')
    
    # Create dataframe for results
    dfResults=pd.DataFrame(columns=['Specialist 1','Specialist 2','Specialist 3','Specialist 4','Specialist 5','Majority vote'], 
                           index=['TP','FP','TN','FN','Sensitivity','Specificity','TPR','FPR','Precision','F1'])
    
    dfResults = AppendResults(dfCombined, dfResults, 'Specialist 1')
    dfResults = AppendResults(dfCombined, dfResults, 'Specialist 2')
    dfResults = AppendResults(dfCombined, dfResults, 'Specialist 3')
    dfResults = AppendResults(dfCombined, dfResults, 'Specialist 4')
    dfResults = AppendResults(dfCombined, dfResults, 'Specialist 5')
    
    dfResults = CalculateResults_Majority(dfCombined, dfResults, 'Majority vote')
    
    return dfResults