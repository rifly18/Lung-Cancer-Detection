def SaveFile(dataframes, filenames):
    import pandas as pd
    for dataframe, filename in zip(dataframes, filenames):
        dataframe.to_pickle(f'{filename}.pkl')
        print(f'file {filename} is saved')