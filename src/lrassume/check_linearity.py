import pandas as pd
from pandas.api.types import is_numeric_dtype

def check_linearity(df: pd.DataFrame, target: str, threshold: float = 0.7) -> pd.DataFrame:
    """Identify features with a specified strength of linear relationship to the target."""
    
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(target, str):
        raise TypeError("Input 'target' must be a string.") 
    if not isinstance(threshold, (int, float)):
        raise TypeError("Input 'threshold' must be a numeric value.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    if not is_numeric_dtype(df[target]):
        raise TypeError("Target column must be numeric.")
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    # Compute correlations for numeric features
    numeric_cols = df.select_dtypes(include='number').columns.drop(target, errors='ignore')
    corrs = df[numeric_cols].corrwith(df[target])

    # Filter correlations by threshold
    corrs = corrs[abs(corrs) >= threshold].round(3)

    # Return empty DataFrame if no features meet threshold
    if corrs.empty:
        return pd.DataFrame({
            "feature": pd.Series(dtype="object"),
            "correlation": pd.Series(dtype="float")
        })

    # Convert to DataFrame and sort by absolute correlation descending, then alphabetically
    result = corrs.reset_index()
    result.columns = ['feature', 'correlation']
    result['abs_corr'] = result['correlation'].abs()
    result = (
        result.sort_values(by=['abs_corr', 'feature'], ascending=[False, True])
              .drop(columns='abs_corr')
              .reset_index(drop=True)
    )

    return result
