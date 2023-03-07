import pandas as pd
from scipy import stats

def Zscore_Individually (df_in, window_cols, all_constant_replace = -1):
    # retriving sets of columns
    cols = df_in.columns.to_list()
    other_cols = list(set(cols) - set(window_cols))
    
    # Zscore can't handle constant (flat) inputs
    # Therefore windows with all constant credit output NaNs
    # Instead these are excluded from zcoring and are manually scaled
    # Default scalling is  const => -1
    # NOTE: this is a simple fix that may lead to issue downstram.
    
    all_constant_replace = float(all_constant_replace)
    # This finds the indexes of the rows with only 1 unique value -> i.e., constant.
    all_constant_index = df_in[df_in[window_cols].nunique(axis=1)==1].index
    var_index = list(set(df_in.index.to_list()) - set(all_constant_index))
    
    if len(all_constant_index.to_list()) > 0:
        # Aux DF for replacing -7 with all_neg_replace (i.e, -1)
        df_allconst = df_in.iloc[all_constant_index].copy()
        df_allconst.loc[:, window_cols] = all_constant_replace

        # Zscoring only non-constant rows
        df_var = df_in.iloc[var_index].copy()    
        df_var[window_cols] = stats.zscore(df_var[window_cols],
                                           axis=1,
                                           nan_policy = 'omit')

        df_zscore = pd.concat([df_var, df_allconst])
        df_zscore.sort_values(by=['short_ID', 'window_ID'], inplace = True)

        # If indeces match, then the join was successful
        if df_zscore.index.to_list() == df_in.index.to_list():
            return df_zscore
        # Otherwise raise error
        else:
            print('Error Zscoring')
    else:
        df_zscore = pd.DataFrame(columns= cols)
        df_zscore[window_cols] = scipy.stats.zscore(df_in[window_cols],
                                                         axis=1,
                                                         nan_policy = 'omit')
        df_zscore[other_cols] = df_in[other_cols]
        

    return df_zscore