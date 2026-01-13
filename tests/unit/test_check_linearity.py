import pytest
import pandas as pd
from lrassume.check_linearity import check_linearity

df_example = pd.DataFrame({
    "sqft": [500, 600, 700, 800, 900],
    "num_rooms": [1, 1, 2, 2, 3],
    "age": [50, 40, 30, 20, 10],
    "school_score": [60, 65, 55, 75, 70],
    "random_noise": [42, 17, 88, 55, 63],
    "price": [150, 180, 210, 240, 270]
})

df_example_2 = df_example[['school_score', 'random_noise', 'price']]

def test_check_linearity_basic():
    """Basic threshold 0.7 with alphabetical tie-breaks"""
    expected = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms'],
        'correlation': [-1.000, 1.000, 0.945]
    })
    result = check_linearity(df_example, target="price", threshold=0.7)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_threshold_one():
    """Test threshold edge case 1.0"""
    expected = pd.DataFrame({
        'feature': ['age', 'sqft'],
        'correlation': [-1.000, 1.000]
    })
    result = check_linearity(df_example, target="price", threshold=1.0)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_threshold_zero():
    """Test threshold 0.0 returns all numeric features"""
    expected = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms', 'school_score', 'random_noise'],
        'correlation': [-1.000, 1.000, 0.945, 0.600, 0.483]
    })
    result = check_linearity(df_example, target="price", threshold=0.0)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_check_linearity_no_features():
    """Test when no features exceed threshold → empty DataFrame"""
    expected = pd.DataFrame({
        'feature': pd.Series(dtype='object'),
        'correlation': pd.Series(dtype='float')
    })
    result = check_linearity(df_example_2, target="price", threshold=0.8)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
