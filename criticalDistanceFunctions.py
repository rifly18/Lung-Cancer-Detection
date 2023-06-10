def PrepareDataForCD(model_names, df1, df2, df3=None, df4=None, df5=None):
    import pandas as pd
    
    precision_lists = []
    recall_lists = []
    f1_lists = []
    specificity_lists = []
    
    # Function to process a single dataframe and add values to the lists
    def process_dataframe(df):
        precision_list = []
        recall_list = []
        f1_list = []
        specificity_list = []
        
        for index, row in df.iterrows():
            if index != 'mean':
                precision_list.append(row['precision'])
                recall_list.append(row['recall'])
                f1_list.append(row['f1'])
                specificity_list.append(row['specificity'])
        
        precision_lists.append(precision_list)
        recall_lists.append(recall_list)
        f1_lists.append(f1_list)
        specificity_lists.append(specificity_list)
    
    # Process each dataframe if provided
    process_dataframe(df1)
    process_dataframe(df2)
    if df3 is not None:
        process_dataframe(df3)
    if df4 is not None:
        process_dataframe(df4)
    if df5 is not None:
        process_dataframe(df5)
    
    # Create a combined dataframe
    data = {'model': [], 'fold': [], 'roc': [], 'specificity': [], 'sensitivity': []}

    for model_name, precision_list, recall_list, f1_list, specificity_list in zip(model_names, precision_lists, recall_lists, f1_lists, specificity_lists):
        for fold in range(1, 6):
            data['model'].append(model_name)
            data['fold'].append(fold)
            data['roc'].append(precision_list[fold-1])
            data['specificity'].append(recall_list[fold-1])
            data['sensitivity'].append(specificity_list[fold-1])

    combined_df = pd.DataFrame(data)

    # reshape data into long format
    data_melted = pd.melt(combined_df, id_vars=['model', 'fold'], var_name='metric', value_name='value')
    
    return combined_df, data_melted

def CalculatePvalues(data, data_melted):
    from scikit_posthocs import posthoc_nemenyi_friedman
    import pandas as pd
    import numpy as np

    # Calculate the Nemenyi test p-values with Friedman's ranking
    p_values = posthoc_nemenyi_friedman(data_melted, y_col='value', block_col='fold', group_col='model', melted=True)

    # Bonferroni correction
    num_comparisons = len(data['model'].unique()) * (len(data['model'].unique()) - 1) / 2
    p_values_bonferroni = p_values * int(num_comparisons)

    return p_values_bonferroni
    
def CriticalDistance_plot(p_values, model_names, fileName):
    from scipy.stats import rankdata
    import numpy as np
    import Orange
    import matplotlib.pyplot as plt
    
    ranks = rankdata(p_values)
    # Reshape the rank matrix so that models are in rows and folds are in columns
    ranks = ranks.reshape(len(model_names), len(model_names)).T
    # Calculate the average rank for each row
    avg_ranks = np.mean(ranks, axis=1)
    # Transpose the resulting vector to get a row vector of average ranks for each model
    avg_ranks = avg_ranks.reshape(1, -1)

    cd = Orange.evaluation.scoring.compute_CD(avg_ranks[0], len(model_names),alpha='0.05', test='nemenyi')
    print(f'Critical distance is: {cd}')

    Orange.evaluation.graph_ranks(avg_ranks[0], model_names, cd=cd, width=6, textspace=0.5)
    plt.savefig(f'{fileName}.pdf', dpi=600, bbox_inches='tight')