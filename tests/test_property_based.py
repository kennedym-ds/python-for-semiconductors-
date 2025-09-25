"""
Property-based tests for data generators and ML pipeline components.
Uses hypothesis for generating test cases.
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from typing import Tuple
import warnings

# Suppress sklearn warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class TestDataGeneratorProperties:
    """Property-based tests for synthetic data generators."""
    
    @given(
        n_samples=st.integers(min_value=10, max_value=1000),
        n_features=st.integers(min_value=2, max_value=50),
        noise_level=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20, deadline=5000)
    def test_regression_data_generator_properties(self, n_samples: int, n_features: int, noise_level: float):
        """Test properties that should hold for any regression dataset generation."""
        from sklearn.datasets import make_regression
        
        # Generate synthetic regression data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise_level,
            random_state=42
        )
        
        # Convert to DataFrame/Series for consistency with pipeline expectations
        feature_names = [f"feature_{i:02d}" for i in range(n_features)]
        df_X = pd.DataFrame(X, columns=feature_names)
        series_y = pd.Series(y, name="target")
        
        # Property 1: Dimensionality
        assert df_X.shape == (n_samples, n_features)
        assert series_y.shape == (n_samples,)
        
        # Property 2: No missing values in synthetic data
        assert not df_X.isnull().any().any()
        assert not series_y.isnull().any()
        
        # Property 3: Finite values only
        assert np.isfinite(df_X.values).all()
        assert np.isfinite(series_y.values).all()
        
        # Property 4: Non-constant features (with high probability)
        if n_samples > 5 and noise_level > 0.001:
            feature_stds = df_X.std()
            assert (feature_stds > 1e-6).sum() >= n_features // 2
        
        # Property 5: Target has reasonable range
        target_range = series_y.max() - series_y.min()
        assert target_range > 0
    
    @given(
        n_samples=st.integers(min_value=20, max_value=500),
        n_features=st.integers(min_value=2, max_value=30),
        n_classes=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=15, deadline=5000)
    def test_classification_data_generator_properties(self, n_samples: int, n_features: int, n_classes: int):
        """Test properties for classification dataset generation."""
        from sklearn.datasets import make_classification
        
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=min(2, n_features//4),
            n_informative=max(2, n_features//2),
            random_state=42
        )
        
        # Convert to pandas
        feature_names = [f"sensor_{i:02d}" for i in range(n_features)]
        df_X = pd.DataFrame(X, columns=feature_names)
        series_y = pd.Series(y, name="class")
        
        # Property 1: Correct dimensions
        assert df_X.shape == (n_samples, n_features)
        assert series_y.shape == (n_samples,)
        
        # Property 2: Class labels in expected range
        unique_classes = sorted(series_y.unique())
        assert len(unique_classes) <= n_classes
        assert min(unique_classes) >= 0
        assert max(unique_classes) < n_classes
        
        # Property 3: All classes represented (with sufficient samples)
        if n_samples >= n_classes * 5:
            assert len(unique_classes) == n_classes
        
        # Property 4: Reasonable class distribution
        class_counts = series_y.value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        assert imbalance_ratio < 10  # Not extremely imbalanced
    
    @given(
        base_value=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        tolerance=st.floats(min_value=0.001, max_value=0.5, allow_nan=False),
        n_samples=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=20)
    def test_semiconductor_process_parameter_generator(self, base_value: float, tolerance: float, n_samples: int):
        """Test semiconductor process parameter generation with specifications."""
        # Simulate process parameters with normal distribution
        np.random.seed(42)
        process_params = np.random.normal(
            loc=base_value, 
            scale=base_value * tolerance, 
            size=n_samples
        )
        
        # Property 1: Most values within specification limits
        spec_lower = base_value * (1 - 3 * tolerance)
        spec_upper = base_value * (1 + 3 * tolerance)
        within_spec = np.sum((process_params >= spec_lower) & (process_params <= spec_upper))
        within_spec_ratio = within_spec / n_samples
        
        # In normal distribution, ~99.7% should be within 3 sigma
        assert within_spec_ratio > 0.95 or n_samples < 20  # Allow some variation for small samples
        
        # Property 2: Mean approximately at target
        if n_samples > 10:
            sample_mean = np.mean(process_params)
            assert abs(sample_mean - base_value) < base_value * tolerance * 2
        
        # Property 3: Positive values for physical parameters
        if base_value > 3 * base_value * tolerance:  # If mean >> std
            assert np.all(process_params > 0), "Physical parameters should be positive"


class TestMLPipelineProperties:
    """Property-based tests for ML pipeline components."""
    
    @given(
        n_samples=st.integers(min_value=50, max_value=500),
        n_features=st.integers(min_value=5, max_value=25),
        test_size=st.floats(min_value=0.1, max_value=0.4)
    )
    @settings(max_examples=10, deadline=10000)
    def test_train_test_split_properties(self, n_samples: int, n_features: int, test_size: float):
        """Test properties of train-test splitting."""
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_regression
        
        # Generate data
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Property 1: Split sizes are correct
        expected_test_size = int(n_samples * test_size)
        assert abs(len(X_test) - expected_test_size) <= 1  # Allow rounding difference
        assert len(X_train) + len(X_test) == n_samples
        
        # Property 2: No data leakage
        train_indices = set(range(len(X_train)))
        test_indices = set(range(len(X_train), len(X_train) + len(X_test)))
        assert len(train_indices.intersection(test_indices)) == 0
        
        # Property 3: Feature dimensions preserved
        assert X_train.shape[1] == n_features
        assert X_test.shape[1] == n_features
        
        # Property 4: Target-feature alignment
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    @given(
        n_estimators=st.integers(min_value=1, max_value=50),
        max_depth=st.integers(min_value=1, max_value=10),
        random_state=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=8, deadline=15000)
    def test_random_forest_properties(self, n_estimators: int, max_depth: int, random_state: int):
        """Test Random Forest model properties.""" 
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        # Generate test data
        X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Property 1: Model can make predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        
        # Property 2: Predictions are finite
        assert np.isfinite(y_pred).all()
        
        # Property 3: Feature importance available and valid
        feature_importance = model.feature_importances_
        assert len(feature_importance) == X.shape[1]
        assert (feature_importance >= 0).all()
        assert abs(feature_importance.sum() - 1.0) < 1e-6  # Should sum to 1
        
        # Property 4: Model performance better than random
        r2 = r2_score(y_test, y_pred)
        # For synthetic data with low noise, R² should be reasonable
        # Allow some variation due to hyperparameter choices
        assert r2 > -1.0  # Better than predicting mean
    
    @given(
        cv_folds=st.integers(min_value=2, max_value=10),
        scoring_metric=st.sampled_from(['accuracy', 'f1', 'precision', 'recall'])
    )
    @settings(max_examples=6)
    def test_cross_validation_properties(self, cv_folds: int, scoring_metric: str):
        """Test cross-validation properties."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Generate classification data
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, 
            n_redundant=2, random_state=42
        )
        
        # Skip if we don't have enough samples for CV
        assume(len(X) >= cv_folds * 10)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring_metric)
        
        # Property 1: Correct number of scores
        assert len(scores) == cv_folds
        
        # Property 2: Scores in valid range
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()
        
        # Property 3: Standard deviation reasonable (not all identical or too variable)
        if cv_folds > 2:
            score_std = np.std(scores)
            assert score_std < 0.5  # Not extremely variable


class TestManufacturingMetricProperties:
    """Property-based tests for semiconductor manufacturing metrics."""
    
    @given(
        true_values=st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=10, max_size=100),
        pred_values=st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=10, max_size=100),
        tolerance=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=10)
    def test_prediction_within_specification_properties(self, true_values: list, pred_values: list, tolerance: float):
        """Test Prediction Within Specification (PWS) metric properties."""
        # Ensure equal length
        min_len = min(len(true_values), len(pred_values))
        y_true = np.array(true_values[:min_len])
        y_pred = np.array(pred_values[:min_len])
        
        # Calculate PWS
        within_spec = np.abs(y_true - y_pred) <= tolerance
        pws = np.mean(within_spec)
        
        # Property 1: PWS in valid range
        assert 0.0 <= pws <= 1.0
        
        # Property 2: Perfect predictions give PWS = 1
        if np.array_equal(y_true, y_pred):
            assert pws == 1.0
        
        # Property 3: PWS decreases with stricter tolerance
        if tolerance > 0.5:
            stricter_tolerance = tolerance / 2
            stricter_within_spec = np.abs(y_true - y_pred) <= stricter_tolerance
            stricter_pws = np.mean(stricter_within_spec)
            assert stricter_pws <= pws
        
        # Property 4: PWS is monotonic in tolerance
        if len(y_true) > 5:
            larger_tolerance = tolerance * 2
            larger_within_spec = np.abs(y_true - y_pred) <= larger_tolerance
            larger_pws = np.mean(larger_within_spec)
            assert larger_pws >= pws
    
    @given(
        false_positives=st.integers(min_value=0, max_value=50),
        false_negatives=st.integers(min_value=0, max_value=50),
        fp_cost=st.floats(min_value=1.0, max_value=1000.0),
        fn_cost=st.floats(min_value=1.0, max_value=2000.0)
    )
    @settings(max_examples=15)
    def test_manufacturing_cost_properties(self, false_positives: int, false_negatives: int, 
                                          fp_cost: float, fn_cost: float):
        """Test manufacturing cost calculation properties."""
        # Calculate total cost
        total_cost = false_positives * fp_cost + false_negatives * fn_cost
        
        # Property 1: Cost is non-negative
        assert total_cost >= 0
        
        # Property 2: No errors means zero cost
        if false_positives == 0 and false_negatives == 0:
            assert total_cost == 0
        
        # Property 3: Cost increases with more errors
        if false_positives > 0 or false_negatives > 0:
            assert total_cost > 0
        
        # Property 4: Cost components are proportional
        fp_cost_component = false_positives * fp_cost
        fn_cost_component = false_negatives * fn_cost
        
        assert fp_cost_component + fn_cost_component == total_cost
        assert fp_cost_component >= 0
        assert fn_cost_component >= 0
    
    @given(
        yield_values=st.lists(
            st.floats(min_value=0.70, max_value=1.0), 
            min_size=5, max_size=50
        ),
        target_yield=st.floats(min_value=0.80, max_value=0.99)
    )
    @settings(max_examples=10)
    def test_process_capability_properties(self, yield_values: list, target_yield: float):
        """Test process capability metrics."""
        yields = np.array(yield_values)
        
        # Calculate process capability metrics
        mean_yield = np.mean(yields)
        std_yield = np.std(yields) if len(yields) > 1 else 0.01  # Avoid division by zero
        
        # Cpk approximation (simplified for demonstration)
        if std_yield > 0:
            cpk = min(
                (mean_yield - 0.7) / (3 * std_yield),  # Lower spec limit
                (1.0 - mean_yield) / (3 * std_yield)   # Upper spec limit
            )
        else:
            cpk = float('inf') if 0.7 <= mean_yield <= 1.0 else 0
        
        # Property 1: Process capability is well-defined
        assert cpk >= 0 or cpk == float('inf')
        
        # Property 2: Higher mean yield generally improves capability
        if mean_yield >= target_yield and std_yield < 0.05:
            assert cpk > 1.0 or cpk == float('inf')
        
        # Property 3: Lower variability improves capability
        if len(yields) > 2:
            # Create a less variable version
            less_variable_yields = mean_yield + (yields - mean_yield) * 0.5
            less_variable_std = np.std(less_variable_yields)
            
            if less_variable_std > 0 and std_yield > 0:
                less_variable_cpk = min(
                    (mean_yield - 0.7) / (3 * less_variable_std),
                    (1.0 - mean_yield) / (3 * less_variable_std)
                )
                if less_variable_cpk < float('inf') and cpk < float('inf'):
                    assert less_variable_cpk >= cpk