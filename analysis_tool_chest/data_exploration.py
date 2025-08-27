import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ConsistencyCheck:
    """
    Bins and aggregates data for a single predictor variable to check consistency/exposure across bins.
    Supports numeric, categorical, and low-cardinality numeric variables.
    
    Parameters:
        data (pd.DataFrame): Input data.
        pred_var (str): Predictor variable to bin.
        exp_var (str): Exposure variable (e.g., vehicle count).
        nbins (int): Number of bins for numeric variables (default 5).
    
    Methods:
        binning(): Performs binning based on pred_var type and stores result in self.binned_data.
        aggregate(year_var): Aggregates exposure and bin statistics by bin and year.
        plot(): Plots exposure percentage by bin and year.
    """
    def __init__(self, data, pred_var, exp_var, nbins=5):
        self.data = data.copy()
        self.pred_var = pred_var
        self.exp_var = exp_var
        self.nbins = nbins
        self.binned_data = None
        self.agg_df = None
        self.year_var = None

    def binning(self):
        pred_var = self.pred_var
        exp_var = self.exp_var
        year_var = self.year_var
        nbins = self.nbins
        df = self.data.copy()

        is_numeric = pd.api.types.is_numeric_dtype(df[pred_var])
        n_unique = df[pred_var].nunique(dropna=True)

        treat_as_cat = is_numeric and n_unique < 20

        if is_numeric and not treat_as_cat:
            # Numeric variable with enough unique values: bin by equal exposure
            missing_mask = df[pred_var].isna()
            df_non_missing = df[~missing_mask].copy()
            df_missing = df[missing_mask].copy()

            df_sorted = df_non_missing.sort_values(pred_var).reset_index(drop=True)
            df_sorted['cum_exp'] = df_sorted[exp_var].cumsum()
            total_exp = df_sorted[exp_var].sum()
            bin_edges_cumexp = [i * total_exp / nbins for i in range(nbins + 1)]

            # Find pred_var values just below each cum_exp bin edge
            pred_var_edges = []
            for edge in bin_edges_cumexp:
                idx = df_sorted[df_sorted['cum_exp'] <= edge].last_valid_index()
                if idx is not None:
                    pred_val = df_sorted.loc[idx, pred_var]
                else:
                    pred_val = df_sorted[pred_var].min()
                pred_var_edges.append(pred_val)
            # Ensure edges are sorted and unique, and cover the full range
            pred_var_edges = sorted(set(pred_var_edges))
            if pred_var_edges[0] > df_sorted[pred_var].min():
                pred_var_edges = [df_sorted[pred_var].min()] + pred_var_edges
            if pred_var_edges[-1] < df_sorted[pred_var].max():
                pred_var_edges.append(df_sorted[pred_var].max())
            # Add a small epsilon to the last edge to include the max value
            pred_var_edges[-1] = pred_var_edges[-1] + 1e-8

            df_sorted['bin'] = pd.cut(df_sorted[pred_var], bins=pred_var_edges, labels=False, include_lowest=True)
            df_sorted = df_sorted.drop(columns=['cum_exp'])

            if not df_missing.empty:
                # Assign missing values to a special bin
                df_missing = df_missing.copy()
                df_missing['bin'] = -1
                df_sorted = pd.concat([df_sorted, df_missing], axis=0, ignore_index=True)

            self.binned_data = df_sorted
        elif treat_as_cat:
            # Numeric variable with low cardinality: treat as categorical
            ordered_cats = sorted(df[pred_var].dropna().unique())
            cat_to_bin = {cat: i for i, cat in enumerate(ordered_cats)}
            df['bin'] = df[pred_var].map(cat_to_bin)
            df.loc[df[pred_var].isna(), 'bin'] = -1
            self.binned_data = df
        else:
            # Categorical variable: use category codes as bins
            ordered_cats = sorted(df[pred_var].dropna().unique())
            df['bin'] = pd.Categorical(df[pred_var], categories=ordered_cats, ordered=True)
            df['bin'] = df['bin'].astype(object)
            df.loc[df[pred_var].isna(), 'bin'] = -1
            self.binned_data = df

    def aggregate(self, year_var):
        pred_var = self.pred_var
        exp_var = self.exp_var
        self.year_var = year_var
        df = self.binned_data.copy()

        is_numeric = pd.api.types.is_numeric_dtype(self.data[pred_var])
        n_unique = self.data[pred_var].nunique(dropna=True)
        treat_as_cat = is_numeric and n_unique < 20

        if is_numeric and not treat_as_cat:
            # Numeric variable with enough unique values: aggregate by bin and year
            agg_df = df.groupby(['bin', year_var], as_index=False).agg({exp_var: 'sum'})
            total_exp_per_year = agg_df.groupby(year_var)[exp_var].transform('sum')
            agg_df[f'{exp_var}_pct'] = agg_df[exp_var] / total_exp_per_year * 100

            bin_stats = df[df['bin'] != -1].groupby(['bin', year_var])[pred_var].agg(['min', 'max']).reset_index()
            bin_stats = bin_stats.rename(columns={'min': f'{pred_var}_min', 'max': f'{pred_var}_max'})
            agg_df = agg_df.merge(bin_stats, on=['bin', year_var], how='left')
        elif treat_as_cat:
            # Numeric variable with low cardinality: aggregate by bin and year, map bin to value
            agg_df = df.groupby(['bin', year_var], as_index=False).agg({exp_var: 'sum'})
            total_exp_per_year = agg_df.groupby(year_var)[exp_var].transform('sum')
            agg_df[f'{exp_var}_pct'] = agg_df[exp_var] / total_exp_per_year * 100
            unique_vals = np.sort(self.data[pred_var].dropna().unique())
            bin_to_val = {i: v for i, v in enumerate(unique_vals)}
            agg_df[f'{pred_var}_cat'] = agg_df['bin'].map(bin_to_val)
        else:
            # Categorical variable: aggregate by bin and year
            agg_df = df.groupby(['bin', year_var], as_index=False).agg({exp_var: 'sum'})
            total_exp_per_year = agg_df.groupby(year_var)[exp_var].transform('sum')
            agg_df[f'{exp_var}_pct'] = agg_df[exp_var] / total_exp_per_year * 100
            agg_df[f'{pred_var}_cat'] = agg_df['bin']
        self.agg_df = agg_df

    def plot(self):
        import matplotlib.pyplot as plt

        pred_var = self.pred_var
        exp_var = self.exp_var
        year_var = self.year_var
        agg_df = self.agg_df

        plt.figure(figsize=(10, 6))

        if f'{pred_var}_max' in agg_df.columns:
            # Numeric variable with enough unique values: plot by bin max
            x_col = f'{pred_var}_max'
            y_col = f'{exp_var}_pct'
            bin_labels = []
            bin_positions = []
            for b in sorted(agg_df['bin'].unique()):
                if b == -1 or b == '-1':
                    bin_labels.append('missing')
                    bin_positions.append(b)
                else:
                    max_val = agg_df.loc[agg_df['bin'] == b, x_col].max()
                    try:
                        label_val = round(float(max_val), 3)
                    except Exception:
                        label_val = max_val
                    bin_labels.append(f"< {label_val}")
                    bin_positions.append(b)
            for year, group in agg_df.groupby(year_var):
                plt.plot(group['bin'], group[y_col], marker='o', label=f'{year_var}: {year}')
            plt.xticks(bin_positions, bin_labels)
            plt.xlabel(f'{pred_var} bin')
            plt.ylabel(y_col + ' (%)')
            plt.title(f'Line Chart of {y_col} by {pred_var} bin and {year_var}')
        else:
            # Categorical or low-cardinality: plot by bin
            x_col = 'bin'
            y_col = f'{exp_var}_pct'
            categories = agg_df[x_col].unique()
            import numpy as np
            cat_labels = np.array([str(c) for c in categories])
            positions = np.arange(len(categories))
            for year, group in agg_df.groupby(year_var):
                group_sorted = group.set_index(x_col).reindex(categories).reset_index()
                plt.plot(positions, group_sorted[y_col], marker='o', label=f'{year_var}: {year}')
            plt.xticks(positions, cat_labels)
            plt.xlabel(pred_var)
            plt.ylabel(y_col + ' (%)')
            plt.title(f'Line Chart of {y_col} by {pred_var} and {year_var}')

        plt.legend(title=year_var)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

class ConsistencyCheckList:
    """
    Runs ConsistencyCheck for a list of predictor variables and provides batch plotting.
    
    Parameters:
        data (pd.DataFrame): Input data.
        pred_var_lst (list of str): List of predictor variables to check.
        expo_var (str): Exposure variable (e.g., vehicle count).
        year_var (str): Year variable for grouping.
        nbins (int): Number of bins for numeric variables (default 5).
    
    Methods:
        run_all(): Runs ConsistencyCheck for each predictor in the list.
        plot_all(): Plots all consistency plots for the predictors.
    """
    def __init__(self, data, pred_var_lst, expo_var, year_var, nbins=5):
        self.data = data
        self.pred_var_lst = pred_var_lst
        self.expo_var = expo_var
        self.year_var = year_var
        self.nbins = nbins
        self.checks = {}  # Store ConsistencyCheck objects by pred_var

    def run_all(self):
    # Run ConsistencyCheck for each predictor variable in the list
        for pred_var in self.pred_var_lst:
            cc = ConsistencyCheck(
                data=self.data,
                pred_var=pred_var,
                exp_var=self.expo_var,
                nbins=self.nbins
            )
            cc.binning()
            cc.aggregate(year_var=self.year_var)
            self.checks[pred_var] = cc

    def plot_all(self):
    # Plot all consistency plots for the predictor variables
        for pred_var, cc in self.checks.items():
            print(f"\n--- Consistency Plot for {pred_var} ---")
            cc.plot()