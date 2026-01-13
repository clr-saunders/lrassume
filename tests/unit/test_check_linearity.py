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

def test_check_linearity():
    """Test check_linearity with example DataFrame and varying thresholds.
        - also tests cases where multiple features have same abs correlation for alphabetical order tie-break
        - edge cases for thresholds 0.0 and 1.0
        - test when correlation equals threshold
        - test threshold below all examples"""
    # Basic Case: 
    # Test with threshold 0.7 and alphabetical order for tie-breaks when abs correlations are equal 
    expected_result_threshold_1 = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms'], 
        'correlation': [-1.000, 1.000, 0.945]  
    })
    actual_result_threshold_1 = check_linearity(df=df_example, target="price", threshold=0.7)
    pd.testing.assert_frame_equal(actual_result_threshold_1.reset_index(drop=True), expected_result_threshold_1)

    # Edge Cases:
    # Test with threshold edge case 1.0 & for when threshold is exactly equal to abs correlation
    expected_result_threshold_2 = pd.DataFrame({
        'feature': ['age', 'sqft'], 
        'correlation': [-1.000, 1.000]  
    })
    actual_result_threshold_2 = check_linearity(df=df_example, target="price", threshold=1.0)
    pd.testing.assert_frame_equal(actual_result_threshold_2.reset_index(drop=True), expected_result_threshold_2)

    # Test with threshold 0.0
    expected_result_threshold_3 = pd.DataFrame({
        'feature': ['age', 'sqft', 'num_rooms', 'school_score', 'random_noise'],
        'correlation': [-1.000, 1.000, 0.945, 0.600, 0.483]})
    actual_result_threshold_3 = check_linearity(df=df_example, target="price", threshold=0.0)
    pd.testing.assert_frame_equal(actual_result_threshold_3.reset_index(drop=True), expected_result_threshold_3)

    # Test threshold above all examples
    expected_result_threshold_4 = pd.D





#def test_check_linearity_errors():

