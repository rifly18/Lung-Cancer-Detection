def NormalizedCM_plot(TN, FP, FN, TP, label, figName):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    sns.set(font_scale=2.0)

    Sum=TN+FP+FN+TP

    CMTest=pd.DataFrame(data=[[(TN/Sum),(FP/Sum)],
                             [(FN/Sum),(TP/Sum)]])

    # Normalized confusion matrix
    conf_mat_norm_test = CMTest.div(CMTest.sum(axis=1), axis=0)*100

    fig=plt.figure()
    fig.set_size_inches(8,7)

    fig = sns.heatmap(conf_mat_norm_test, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'format': '%.0f'});

    fig.set_xlabel(f'\n{label}')
    fig.set_ylabel('Actual label');

    fig.xaxis.set_ticklabels(['False','True'])
    fig.yaxis.set_ticklabels(['False','True'])
    fig.xaxis.set_ticklabels(['Non-LC','LC']);
    fig.yaxis.set_ticklabels(['Non-LC','LC']);
    
    plt.savefig(f'{figName}.pdf', dpi=600, bbox_inches='tight')

def plot_specialist_results_roc(dfResults, specialistName, color=None, marker=None, markersize=None, label=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    if color is None:
        color = "red"
    if marker is None:
        marker = "+"
    if markersize is None:
        markersize = 14
    
    line = plt.plot(
        dfResults[specialistName].loc['FPR'],
        dfResults[specialistName].loc['TPR'],
        color=color,
        marker=marker,
        markersize=markersize,
        label=label
    )
    return line

# Only call with both input dataframes when plotting the 200 samples
def ROC_plot(dfROC, fileName, dfSpecialists=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    #Interpolate between datapoints
    mean_fpr = np.linspace(0, 1, 100)
    tprs_interp = []
    for each in range(len(dfROC['TPR'])):
        tpr_interp = np.interp(mean_fpr, dfROC['FPR'][each], dfROC['TPR'][each])
        tprs_interp.append(tpr_interp)

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    mean_auc=np.mean(dfROC['auc'])
    sd=np.std(dfROC['auc'])

    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr+ 2*std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 2*std_tpr, 0)
    
    #Create plot
    sns.set(font_scale=2.0)
    sns.set_style('ticks')

    plt.figure(figsize=(8,7))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, sd),
        lw=2,
        alpha=0.8)

    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="red",
        alpha=0.2,
        label=r"$\pm$ 2 standard deviation")
    
    if dfSpecialists is not None:
        line2, = plot_specialist_results_roc(dfSpecialists, 'Specialist 1', label = "Individual specialist")
        plot_specialist_results_roc(dfSpecialists, 'Specialist 2')
        plot_specialist_results_roc(dfSpecialists, 'Specialist 3')
        plot_specialist_results_roc(dfSpecialists, 'Specialist 4')
        plot_specialist_results_roc(dfSpecialists, 'Specialist 5')
        line3, = plot_specialist_results_roc(dfSpecialists, 'Majority vote', color="green", marker="o", markersize=12, label = "Average specialist")

        line2.set_linestyle('None')
        line3.set_linestyle('None')
        legend=plt.legend(loc="lower right", fontsize=18)
        
    else:
        plt.legend(loc="lower right", fontsize=20)
    
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    
    plt.savefig(f'{fileName}.pdf', dpi=600, bbox_inches='tight')

def plot_specialist_results_pr(dfResults, specialistName, color=None, marker=None, markersize=None, label=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    if color is None:
        color = "red"
    if marker is None:
        marker = "+"
    if markersize is None:
        markersize = 14
    
    line = plt.plot(
        dfResults[specialistName].loc['Sensitivity'],
        dfResults[specialistName].loc['Precision'],
        color=color,
        marker=marker,
        markersize=markersize,
        label=label
    )
    
    return line

# Only call with both input dataframes when plotting the 200 samples
def PR_plot(dfPR, fileName, dfSpecialists=None):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Lengths of dataframes
    precision_lengths = [len(lst) for lst in dfPR['precision']]   
    recall_lengths = [len(lst) for lst in dfPR['recall']]

    # Find the minimum length
    min_length = min(precision_lengths + recall_lengths)
    
    #Make sure that all folds have an equal length
    for each in range(len(dfPR['precision'])):
        dfPR['precision'][each] = dfPR['precision'][each][:min_length]
    for each in range(len(dfPR['recall'])):
        dfPR['recall'][each] = dfPR['recall'][each][:min_length]

    mean_precision=np.mean(dfPR['precision'],axis=0)
    mean_recall=np.mean(dfPR['recall'],axis=0)
    mean_auc=np.mean(dfPR['auc'])
    sd=np.std(dfPR['auc'])

    std_tpr = np.std(np.array(dfPR['precision']), axis=0)
    tprs_upper = np.minimum(mean_precision+ 2*std_tpr, 1)
    tprs_lower = np.maximum(mean_precision - 2*std_tpr, 0)
    
    # Plotting
    sns.set(font_scale=2.0)
    sns.set_style('ticks')

    plt.figure(figsize=(8,7))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.plot(
        mean_recall,
        mean_precision,
        color="blue",
        #label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, sd),
        lw=2,
        alpha=0.8,
    )

    # Add standard deviation
    '''
    plt.fill_between(
        mean_recall,
        tprs_lower,
        tprs_upper,
        color="red",
        alpha=0.2,
        label=r"$\pm$ 2 std. dev.",
    )
    '''
    if dfSpecialists is not None:
        line2, = plot_specialist_results_pr(dfSpecialists, 'Specialist 1', label = "Individual specialist")
        plot_specialist_results_pr(dfSpecialists, 'Specialist 2')
        plot_specialist_results_pr(dfSpecialists, 'Specialist 3')
        plot_specialist_results_pr(dfSpecialists, 'Specialist 4')
        plot_specialist_results_pr(dfSpecialists, 'Specialist 5')
        line3, = plot_specialist_results_pr(dfSpecialists, 'Majority vote', color="green", marker="o", markersize=12, label = "Average specialist")

        line2.set_linestyle('None')
        line3.set_linestyle('None')
        legend=plt.legend(loc="lower right", fontsize=18)
        
    else:
        plt.legend(loc="lower right", fontsize=20)
    
    plt.xlabel("Sensitivity (Recall)")
    plt.ylabel("PPV (Precision)")
    
    plt.savefig(f'{fileName}.pdf', dpi=600, bbox_inches='tight')

def Preds_plot(dfPreds, fileName):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.set(font_scale=1.8)    
    sns.set_style('ticks')
    fig, ax1 = plt.subplots(figsize=(8, 8))

    bin_width = 0.1
    bins = np.arange(0, 1.1, bin_width)

    hist, _ = np.histogram(dfPreds['probability'][0], bins=bins)
    
    width = 0.05  # Width of each bar
    offset = 0.05  # Offset between the two bars

    ax1.bar(bins[:-1], hist, width=width, align='edge', color='silver', label='Prediction') 
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Predictions [Counts]', color='silver')
    ax1.set_xticks(np.arange(0, 1.1, 0.1))

    y_true_counts = []
    y_pred_counts = []
    for i in range(len(bins)-1):
        y_true_bin = dfPreds['actual'][0][(dfPreds['probability'][0] >= bins[i]) & (dfPreds['probability'][0] < bins[i+1])]
        y_true_1_count = np.sum(y_true_bin == 1)
        y_true_counts.append(y_true_1_count)
        y_pred_bin = dfPreds['probability'][0][(dfPreds['probability'][0] >= bins[i]) & (dfPreds['probability'][0] < bins[i+1])]
        y_pred_counts.append(len(y_pred_bin))
    
    percentages = []
    for i in range(len(bins)-1):
        true_count = y_true_counts[i]
        pred_count = y_pred_counts[i]
        if pred_count == 0:
            percentages.append(0)
        else:
            percentages.append((true_count / pred_count) * 100)
    
    ax2 = ax1.twinx()
    ax2.bar(bins[:-1] + offset, percentages, width=width, align='edge', color='sandybrown', label='LC incidence')  # Second bar with offset
    ax2.set_ylabel('LC incidence [%]', color='sandybrown')
    
    ax1.tick_params(axis='y', colors='silver')
    ax2.tick_params(axis='y', colors='sandybrown')
    
    plt.savefig(f'{fileName}.pdf', dpi=600, bbox_inches='tight')