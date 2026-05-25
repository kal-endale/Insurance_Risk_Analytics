import pandas as pd
from scipy.stats import chi2_contingency, f_oneway

def chi_squared_test_for_claim_frequency(df: pd.DataFrame, group_col: str) -> dict:
    """
    Performs a chi-squared test for independence on claim frequency across groups.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'Claimed' and the grouping column.
        group_col (str): The name of the column to group by (e.g., 'Province', 'Gender').

    Returns:
        dict: A dictionary containing the chi-squared statistic, p-value, and degrees of freedom.
    """
    contingency_table = pd.crosstab(df[group_col], df['Claimed'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return {
        'statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'contingency_table': contingency_table
    }

def anova_test_for_claim_severity(df: pd.DataFrame, group_col: str) -> dict:
    """
    Performs an ANOVA test on claim severity across groups.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'ClaimAmount' and the grouping column.
                           Only rows where 'Claimed' is True are considered.
        group_col (str): The name of the column to group by (e.g., 'Province', 'Gender').

    Returns:
        dict: A dictionary containing the F-statistic and p-value.
              Returns None if there are fewer than two groups with claims.
    """
    # Filter for policies with claims
    df_claimed = df[df['Claimed'] == True]

    # Get claim amounts for each group
    groups_data = [group['ClaimAmount'].values for name, group in df_claimed.groupby(group_col)]

    if len(groups_data) < 2:
        return {'statistic': None, 'p_value': None, 'error': 'Not enough groups with claims for ANOVA'}

    # Remove empty arrays which can occur if a group has no claims
    groups_data = [g for g in groups_data if len(g) > 0]

    if len(groups_data) < 2:
        return {'statistic': None, 'p_value': None, 'error': 'Not enough groups with claims for ANOVA after filtering empty'}

    f_statistic, p_value = f_oneway(*groups_data)
    return {
        'statistic': f_statistic,
        'p_value': p_value
    }
