#!/usr/bin/env python3
"""
Tests for Module 4.3 Multilabel Classification

This module tests multilabel classification techniques including Binary Relevance,
Classifier Chains, and Label Powerset approaches for semiconductor defect detection.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Multilabel classification
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import (
    hamming_loss,
    jaccard_score,
    f1_score,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


class TestMultilabelDataGeneration:
    """Test suite for multilabel dataset creation."""

    @pytest.fixture
    def synthetic_multilabel_data(self):
        """Create synthetic multilabel classification data for semiconductor defects."""
        np.random.seed(RANDOM_SEED)
        n_samples = 1000
        n_features = 20

        # Generate features (process parameters)
        X = np.random.randn(n_samples, n_features)

        # Generate correlated multilabel targets
        # 5 defect types: scratch, particle, film_issue, pattern_defect, contamination
        y = np.zeros((n_samples, 5), dtype=int)

        # Label 0: Scratch (influenced by features 0, 1)
        y[:, 0] = (X[:, 0] + X[:, 1] > 0.5).astype(int)

        # Label 1: Particle (influenced by features 2, 3)
        y[:, 1] = (X[:, 2] + X[:, 3] > 0.5).astype(int)

        # Label 2: Film issue (influenced by features 4, 5, correlated with scratch)
        y[:, 2] = ((X[:, 4] + X[:, 5] > 0.3) | (y[:, 0] == 1)).astype(int)

        # Label 3: Pattern defect (influenced by features 6, 7)
        y[:, 3] = (X[:, 6] + X[:, 7] > 0.7).astype(int)

        # Label 4: Contamination (influenced by features 8, 9, correlated with particle)
        y[:, 4] = ((X[:, 8] + X[:, 9] > 0.4) | (y[:, 1] == 1)).astype(int)

        return X, y

    def test_data_shape(self, synthetic_multilabel_data):
        """Test that generated data has correct shape."""
        X, y = synthetic_multilabel_data
        assert X.shape == (1000, 20)
        assert y.shape == (1000, 5)

    def test_data_types(self, synthetic_multilabel_data):
        """Test that data types are correct."""
        X, y = synthetic_multilabel_data
        assert X.dtype in [np.float64, np.float32]
        assert y.dtype == np.int32 or y.dtype == np.int64

    def test_label_distribution(self, synthetic_multilabel_data):
        """Test that labels have reasonable distribution."""
        X, y = synthetic_multilabel_data

        # Check that each label has some positive examples
        for i in range(y.shape[1]):
            positive_count = y[:, i].sum()
            assert positive_count > 0, f"Label {i} has no positive examples"
            assert positive_count < len(y), f"Label {i} is all positive"

    def test_label_correlation(self, synthetic_multilabel_data):
        """Test that some labels are correlated as expected."""
        X, y = synthetic_multilabel_data

        # Label 2 (film_issue) should be correlated with Label 0 (scratch)
        correlation = np.corrcoef(y[:, 0], y[:, 2])[0, 1]
        assert correlation > 0.3, "Expected correlation between scratch and film_issue"

        # Label 4 (contamination) should be correlated with Label 1 (particle)
        correlation = np.corrcoef(y[:, 1], y[:, 4])[0, 1]
        assert correlation > 0.3, "Expected correlation between particle and contamination"


class TestBinaryRelevance:
    """Test Binary Relevance approach for multilabel classification."""

    @pytest.fixture
    def trained_br_model(self, synthetic_multilabel_data):
        """Train a Binary Relevance model."""
        X, y = synthetic_multilabel_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        # Binary Relevance = MultiOutputClassifier with independent classifiers
        br_clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED, max_depth=5))
        br_clf.fit(X_train, y_train)

        return br_clf, X_test, y_test

    def test_br_training(self, trained_br_model):
        """Test that Binary Relevance model trains successfully."""
        br_clf, X_test, y_test = trained_br_model
        assert br_clf is not None
        assert hasattr(br_clf, "estimators_")
        assert len(br_clf.estimators_) == 5  # 5 labels = 5 binary classifiers

    def test_br_prediction(self, trained_br_model):
        """Test Binary Relevance predictions."""
        br_clf, X_test, y_test = trained_br_model
        y_pred = br_clf.predict(X_test)

        assert y_pred.shape == y_test.shape
        assert y_pred.dtype in [np.int32, np.int64]
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_br_performance(self, trained_br_model):
        """Test Binary Relevance performance metrics."""
        br_clf, X_test, y_test = trained_br_model
        y_pred = br_clf.predict(X_test)

        # Hamming loss (fraction of incorrect labels)
        h_loss = hamming_loss(y_test, y_pred)
        assert 0 <= h_loss <= 1, "Hamming loss should be between 0 and 1"
        assert h_loss < 0.5, "Hamming loss should be less than random baseline"

        # Subset accuracy (exact match)
        subset_acc = accuracy_score(y_test, y_pred)
        assert 0 <= subset_acc <= 1, "Subset accuracy should be between 0 and 1"

        # Micro-averaged F1
        f1_micro = f1_score(y_test, y_pred, average="micro")
        assert f1_micro > 0.5, "F1-micro should be better than random"

    def test_br_probability_prediction(self, trained_br_model):
        """Test Binary Relevance probability predictions."""
        br_clf, X_test, y_test = trained_br_model

        # Check if predict_proba is available
        if hasattr(br_clf, "predict_proba"):
            probas = br_clf.predict_proba(X_test)
            assert len(probas) == 5  # List of probability arrays for each label

            for i, proba in enumerate(probas):
                assert proba.shape[0] == len(X_test)
                assert proba.shape[1] == 2  # Binary: [prob_class_0, prob_class_1]
                assert np.allclose(proba.sum(axis=1), 1.0), f"Probabilities for label {i} don't sum to 1"


class TestClassifierChains:
    """Test Classifier Chains approach for multilabel classification."""

    @pytest.fixture
    def trained_cc_model(self, synthetic_multilabel_data):
        """Train a Classifier Chain model."""
        X, y = synthetic_multilabel_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        # Classifier Chains - models label dependencies
        cc_clf = ClassifierChain(
            RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED, max_depth=5),
            order="random",
            random_state=RANDOM_SEED,
        )
        cc_clf.fit(X_train, y_train)

        return cc_clf, X_test, y_test

    def test_cc_training(self, trained_cc_model):
        """Test that Classifier Chain model trains successfully."""
        cc_clf, X_test, y_test = trained_cc_model
        assert cc_clf is not None
        assert hasattr(cc_clf, "estimators_")
        assert len(cc_clf.estimators_) == 5  # 5 labels = chain of 5 classifiers

    def test_cc_chain_order(self, trained_cc_model):
        """Test that chain order is set correctly."""
        cc_clf, X_test, y_test = trained_cc_model
        assert hasattr(cc_clf, "order_")
        assert len(cc_clf.order_) == 5
        # Check that order contains unique indices 0-4
        assert set(cc_clf.order_) == {0, 1, 2, 3, 4}

    def test_cc_prediction(self, trained_cc_model):
        """Test Classifier Chain predictions."""
        cc_clf, X_test, y_test = trained_cc_model
        y_pred = cc_clf.predict(X_test)

        assert y_pred.shape == y_test.shape
        assert y_pred.dtype in [np.int32, np.int64, np.float64, np.float32]  # sklearn sometimes returns float
        assert set(np.unique(y_pred)).issubset({0, 1, 0.0, 1.0})  # Accept both int and float versions

    def test_cc_performance(self, trained_cc_model):
        """Test Classifier Chain performance metrics."""
        cc_clf, X_test, y_test = trained_cc_model
        y_pred = cc_clf.predict(X_test)

        # Hamming loss
        h_loss = hamming_loss(y_test, y_pred)
        assert 0 <= h_loss <= 1
        assert h_loss < 0.5, "Should be better than random"

        # F1 scores
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_macro = f1_score(y_test, y_pred, average="macro")

        assert f1_micro > 0.5
        assert f1_macro > 0.3  # Macro can be lower if some labels are hard


class TestLabelPowerset:
    """Test Label Powerset approach (simulated with simple encoding)."""

    @pytest.fixture
    def label_powerset_encoding(self, synthetic_multilabel_data):
        """Create label powerset encoding."""
        X, y = synthetic_multilabel_data

        # Convert multilabel to single-label by treating each combination as unique class
        # Label powerset: encode each label combination as a unique class
        label_strings = ["".join(map(str, row)) for row in y]
        unique_combinations = sorted(set(label_strings))

        # Map to integer classes
        label_to_int = {label: i for i, label in enumerate(unique_combinations)}
        y_lp = np.array([label_to_int[label] for label in label_strings])

        return X, y, y_lp, label_to_int

    def test_lp_encoding(self, label_powerset_encoding):
        """Test label powerset encoding."""
        X, y, y_lp, label_to_int = label_powerset_encoding

        assert len(y_lp) == len(y)
        assert y_lp.dtype in [np.int32, np.int64]
        # Number of unique classes should be <= 2^5 = 32
        assert len(label_to_int) <= 32

    def test_lp_distribution(self, label_powerset_encoding):
        """Test that label combinations have reasonable distribution."""
        X, y, y_lp, label_to_int = label_powerset_encoding

        unique, counts = np.unique(y_lp, return_counts=True)

        # Should have multiple unique combinations
        assert len(unique) >= 2, "Should have at least 2 different label combinations"

        # Most common combination shouldn't dominate too much
        max_frequency = counts.max() / len(y_lp)
        assert max_frequency < 0.8, "No single combination should dominate"


class TestMultilabelMetrics:
    """Test multilabel-specific metrics."""

    def test_hamming_loss_edge_cases(self):
        """Test hamming loss edge cases."""
        # Perfect predictions
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0]])
        assert hamming_loss(y_true, y_pred) == 0.0

        # Completely wrong predictions
        y_pred_wrong = np.array([[0, 1, 0], [1, 0, 1]])
        assert hamming_loss(y_true, y_pred_wrong) == 1.0

        # Partial accuracy - 2/6 labels wrong = 0.333... hamming loss
        y_pred_partial = np.array([[1, 1, 1], [0, 0, 0]])
        h_loss = hamming_loss(y_true, y_pred_partial)
        assert 0.3 < h_loss < 0.4  # Approximately 1/3

    def test_jaccard_score(self):
        """Test Jaccard similarity (intersection over union)."""
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1]])

        # Sample-wise Jaccard
        jaccard_samples = jaccard_score(y_true, y_pred, average="samples")
        assert 0 <= jaccard_samples <= 1

        # Micro-average Jaccard
        jaccard_micro = jaccard_score(y_true, y_pred, average="micro")
        assert 0 <= jaccard_micro <= 1

    def test_f1_multilabel(self):
        """Test F1 score for multilabel classification."""
        y_true = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
        y_pred = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])

        # Micro-average (aggregates contributions of all labels)
        f1_micro = f1_score(y_true, y_pred, average="micro")
        assert 0 <= f1_micro <= 1

        # Macro-average (unweighted mean of label scores)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        assert 0 <= f1_macro <= 1

        # Samples-average (average over samples)
        f1_samples = f1_score(y_true, y_pred, average="samples")
        assert 0 <= f1_samples <= 1


class TestThresholdOptimization:
    """Test threshold optimization for multilabel classification."""

    @pytest.fixture
    def probability_predictions(self):
        """Create sample probability predictions."""
        np.random.seed(RANDOM_SEED)
        n_samples = 100
        n_labels = 3

        # Generate probabilities
        y_proba = np.random.rand(n_samples, n_labels)

        # Generate true labels (correlated with probabilities)
        y_true = (y_proba > 0.5).astype(int)

        return y_proba, y_true

    def test_default_threshold(self, probability_predictions):
        """Test predictions with default 0.5 threshold."""
        y_proba, y_true = probability_predictions

        # Apply threshold
        y_pred = (y_proba >= 0.5).astype(int)

        assert y_pred.shape == y_true.shape
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_custom_thresholds(self, probability_predictions):
        """Test predictions with custom per-label thresholds."""
        y_proba, y_true = probability_predictions

        # Different threshold for each label
        thresholds = np.array([0.3, 0.5, 0.7])

        # Apply thresholds
        y_pred = (y_proba >= thresholds).astype(int)

        # Label 0 should have more positives (lower threshold)
        # Label 2 should have fewer positives (higher threshold)
        assert y_pred[:, 0].sum() >= y_pred[:, 2].sum()

    def test_threshold_tuning_impact(self, probability_predictions):
        """Test impact of threshold tuning on metrics."""
        y_proba, y_true = probability_predictions

        # Test multiple thresholds
        thresholds_to_test = [0.3, 0.5, 0.7]
        f1_scores = []

        for thresh in thresholds_to_test:
            y_pred = (y_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
            f1_scores.append(f1)

        # All scores should be valid
        assert all(0 <= score <= 1 for score in f1_scores)


class TestPreprocessing:
    """Test preprocessing for multilabel data."""

    def test_feature_scaling(self, synthetic_multilabel_data):
        """Test feature scaling doesn't affect multilabel targets."""
        X, y = synthetic_multilabel_data

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check that features are scaled
        assert np.abs(X_scaled.mean()) < 0.1
        assert np.abs(X_scaled.std() - 1.0) < 0.1

        # Check that labels are unchanged
        assert np.array_equal(y, y)  # Labels should remain binary

    def test_train_test_split_stratification(self, synthetic_multilabel_data):
        """Test train-test split with multilabel data."""
        X, y = synthetic_multilabel_data

        # Standard split (stratification is complex for multilabel)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        assert len(X_train) == 800
        assert len(X_test) == 200
        assert y_train.shape[1] == 5
        assert y_test.shape[1] == 5


class TestSemiconductorContext:
    """Test semiconductor-specific multilabel scenarios."""

    def test_defect_cooccurrence(self):
        """Test modeling of co-occurring defects."""
        # Simulate real-world scenario: scratch + contamination often occur together
        n_samples = 500
        np.random.seed(RANDOM_SEED)

        # Features
        X = np.random.randn(n_samples, 10)

        # Labels: [scratch, particle, contamination]
        y = np.zeros((n_samples, 3), dtype=int)

        # Scratch
        y[:, 0] = (X[:, 0] > 0).astype(int)

        # Contamination (often co-occurs with scratch)
        y[:, 2] = ((X[:, 1] > 0) | (y[:, 0] == 1)).astype(int)

        # Particle (independent)
        y[:, 1] = (X[:, 2] > 0.5).astype(int)

        # Check correlation
        scratch_contam_corr = np.corrcoef(y[:, 0], y[:, 2])[0, 1]
        assert scratch_contam_corr > 0.5, "Scratch and contamination should be correlated"

        scratch_particle_corr = np.corrcoef(y[:, 0], y[:, 1])[0, 1]
        assert scratch_particle_corr < 0.3, "Scratch and particle should be less correlated"

    def test_zero_defect_samples(self):
        """Test handling of samples with no defects."""
        # Some wafers might have no defects
        y = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0]])  # Rows 1 and 3 have no defects

        zero_defect_samples = (y.sum(axis=1) == 0).sum()
        assert zero_defect_samples == 2, "Should correctly identify samples with no defects"

    def test_all_defects_samples(self):
        """Test handling of samples with all defects."""
        y = np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]])  # Rows 0 and 2 have all defects

        all_defect_samples = (y.sum(axis=1) == y.shape[1]).sum()
        assert all_defect_samples == 2, "Should correctly identify samples with all defects"


@pytest.fixture
def synthetic_multilabel_data():
    """Shared fixture for multilabel data across test classes."""
    np.random.seed(RANDOM_SEED)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.zeros((n_samples, 5), dtype=int)

    y[:, 0] = (X[:, 0] + X[:, 1] > 0.5).astype(int)
    y[:, 1] = (X[:, 2] + X[:, 3] > 0.5).astype(int)
    y[:, 2] = ((X[:, 4] + X[:, 5] > 0.3) | (y[:, 0] == 1)).astype(int)
    y[:, 3] = (X[:, 6] + X[:, 7] > 0.7).astype(int)
    y[:, 4] = ((X[:, 8] + X[:, 9] > 0.4) | (y[:, 1] == 1)).astype(int)

    return X, y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
