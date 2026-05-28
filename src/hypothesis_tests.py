import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind


# =========================================================
# 1. FEATURE ENGINEERING
# =========================================================

def create_risk_metrics(df):
    """
    Create insurance risk metrics.

    Returns:
        DataFrame with:
        - HasClaim
        - ClaimSeverity
        - Margin
    """

    df = df.copy()

    # Claim Frequency Indicator
    df['HasClaim'] = np.where(df['totalclaims'] > 0, 1, 0)

    # Claim Severity
    df['ClaimSeverity'] = np.where(
        df['totalclaims'] > 0,
        df['totalclaims'],
        np.nan
    )

    # Margin
    df['Margin'] = df['totalpremium'] - df['totalclaims']

    return df


# =========================================================
# 2. DATA SEGMENTATION
# =========================================================

def segment_data(df, column_name, group_a, group_b):
    """
    Create control and test groups.

    Parameters:
    -----------
    column_name : str
        Feature used for segmentation

    group_a : str/int
        Control group

    group_b : str/int
        Test group
    """

    segmented_df = df[
        df[column_name].isin([group_a, group_b])
    ].copy()

    return segmented_df


# =========================================================
# 3. STATISTICAL EQUIVALENCE CHECKS
# =========================================================

def compare_categorical_distribution(df, group_col, compare_col):
    """
    Compare categorical distributions between groups.
    """

    table = pd.crosstab(
        df[group_col],
        df[compare_col],
        normalize='index'
    )

    return table


def compare_numerical_distribution(df, group_col, numeric_col):
    """
    Compare numerical feature distributions.
    """

    summary = df.groupby(group_col)[numeric_col].describe()

    return summary


# =========================================================
# 4. CHI-SQUARE TEST
# =========================================================

def chi_square_test(df, group_col, target_col):
    """
    Perform Chi-Square test.

    Used for:
    - Claim Frequency
    - Binary/Categorical KPIs
    """

    contingency_table = pd.crosstab(
        df[group_col],
        df[target_col]
    )

    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    return {
        'test': 'Chi-Square',
        'p_value': p_value,
        'chi2_statistic': chi2,
        'degrees_of_freedom': dof
    }


# =========================================================
# 5. T-TEST
# =========================================================

def t_test(
    df,
    group_col,
    target_col,
    group_a,
    group_b
):
    """
    Perform independent t-test.

    Used for:
    - Claim Severity
    - Margin
    """

    sample_a = df[
        df[group_col] == group_a
    ][target_col].dropna()

    sample_b = df[
        df[group_col] == group_b
    ][target_col].dropna()

    t_stat, p_value = ttest_ind(
        sample_a,
        sample_b,
        equal_var=False
    )

    return {
        'test': 'Independent T-Test',
        'p_value': p_value,
        't_statistic': t_stat,
        'group_a_mean': sample_a.mean(),
        'group_b_mean': sample_b.mean()
    }


# =========================================================
# 6. HYPOTHESIS DECISION
# =========================================================

def hypothesis_decision(p_value, alpha=0.05):
    """
    Decide whether to reject H0.
    """

    if p_value < alpha:
        return "Reject H0"
    else:
        return "Fail to Reject H0"


# =========================================================
# 7. BUSINESS INTERPRETATION
# =========================================================

def business_interpretation(
    hypothesis_name,
    p_value,
    decision,
    metric_name,
    group_a,
    group_b,
    group_a_mean=None,
    group_b_mean=None
):
    """
    Generate business-friendly interpretation.
    """

    if decision == "Reject H0":

        if (
            group_a_mean is not None and
            group_b_mean is not None
        ):

            difference = (
                (group_b_mean - group_a_mean)
                / group_a_mean
            ) * 100

            interpretation = (
                f"We reject H0 for {hypothesis_name} "
                f"(p = {p_value:.4f}). "
                f"{group_b} exhibits a "
                f"{difference:.2f}% difference in "
                f"{metric_name} compared to {group_a}. "
                f"This suggests a business-relevant "
                f"risk difference that may justify "
                f"pricing or underwriting adjustments."
            )

        else:

            interpretation = (
                f"We reject H0 for {hypothesis_name} "
                f"(p = {p_value:.4f}). "
                f"There is a statistically significant "
                f"difference between {group_a} and "
                f"{group_b}."
            )

    else:

        interpretation = (
            f"We fail to reject H0 for "
            f"{hypothesis_name} "
            f"(p = {p_value:.4f}). "
            f"No statistically significant difference "
            f"was detected between {group_a} and "
            f"{group_b}."
        )

    return interpretation


# =========================================================
# 8. FULL HYPOTHESIS PIPELINE
# =========================================================

def run_hypothesis_test(
    df,
    hypothesis_name,
    group_col,
    target_col,
    group_a,
    group_b,
    test_type='chi_square'
):
    """
    Complete hypothesis testing pipeline.
    """

    # Segment data
    segmented_df = segment_data(
        df,
        group_col,
        group_a,
        group_b
    )

    # Run statistical test
    if test_type == 'chi_square':

        results = chi_square_test(
            segmented_df,
            group_col,
            target_col
        )

    elif test_type == 't_test':

        results = t_test(
            segmented_df,
            group_col,
            target_col,
            group_a,
            group_b
        )

    else:
        raise ValueError(
            "Invalid test type. "
            "Choose 'chi_square' or 't_test'."
        )

    # Decision
    decision = hypothesis_decision(
        results['p_value']
    )

    # Interpretation
    interpretation = business_interpretation(
        hypothesis_name=hypothesis_name,
        p_value=results['p_value'],
        decision=decision,
        metric_name=target_col,
        group_a=group_a,
        group_b=group_b,
        group_a_mean=results.get('group_a_mean'),
        group_b_mean=results.get('group_b_mean')
    )

    # Final output
    final_results = {
        'Hypothesis': hypothesis_name,
        'Test': results['test'],
        'P-Value': results['p_value'],
        'Decision': decision,
        'Interpretation': interpretation
    }

    return final_results