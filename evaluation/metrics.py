"""
Evaluation Metrics Module

Comprehensive metrics for binary classification of antibody-antigen interactions.

Primary metric: AUROC (from paper)
Secondary metrics: AUPRC, Accuracy, F1, MCC

Paper benchmarks to match/beat:
- Lenient: AUROC 0.91-0.92
- HA-exclusive: AUROC 0.90
- mAb-exclusive: AUROC 0.73 (key challenge)
- mAb-cluster: AUROC 0.63-0.66 (hardest)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)


# Paper benchmark values
PAPER_BENCHMARKS = {
    'binding': {
        'lenient': {'auroc': 0.92, 'std': 0.02},
        'ha_exclusive': {'auroc': 0.90, 'std': 0.03},
        'mab_exclusive': {'auroc': 0.73, 'std': 0.05},
        'mab_cluster': {'auroc': 0.66, 'std': 0.06}
    },
    'hai': {
        'lenient': {'auroc': 0.91, 'std': 0.03},
        'ha_exclusive': {'auroc': 0.90, 'std': 0.04},
        'mab_exclusive': {'auroc': 0.73, 'std': 0.06},
        'mab_cluster': {'auroc': 0.63, 'std': 0.07}
    }
}


@dataclass
class BinaryClassificationMetrics:
    """Container for binary classification metrics.
    
    Attributes:
        auroc: Area Under ROC Curve (primary metric)
        auprc: Area Under Precision-Recall Curve
        accuracy: Classification accuracy
        precision: Precision (positive predictive value)
        recall: Recall (sensitivity)
        f1: F1 score
        mcc: Matthews Correlation Coefficient
        specificity: Specificity (true negative rate)
        threshold: Optimal classification threshold
        n_samples: Number of samples
        n_positive: Number of positive samples
        n_negative: Number of negative samples
    """
    auroc: float = 0.0
    auprc: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mcc: float = 0.0
    specificity: float = 0.0
    threshold: float = 0.5
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    
    # Optional: confidence intervals
    auroc_ci: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'auroc': self.auroc,
            'auprc': self.auprc,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'mcc': self.mcc,
            'specificity': self.specificity,
            'threshold': self.threshold,
            'n_samples': self.n_samples,
            'n_positive': self.n_positive,
            'n_negative': self.n_negative
        }
        
    def summary(self) -> str:
        """Return formatted summary string."""
        return (
            f"AUROC: {self.auroc:.4f} | AUPRC: {self.auprc:.4f} | "
            f"Acc: {self.accuracy:.4f} | F1: {self.f1:.4f} | "
            f"MCC: {self.mcc:.4f} | n={self.n_samples} ({self.n_positive}+/{self.n_negative}-)"
        )


def compute_auroc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    return_curve: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """Compute Area Under ROC Curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities or scores
        return_curve: If True, also return FPR and TPR arrays
        
    Returns:
        AUROC value, optionally (auroc, fpr, tpr)
    """
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class in y_true. AUROC undefined.")
            return 0.5 if not return_curve else (0.5, None, None)
            
        auroc = roc_auc_score(y_true, y_scores)
        
        if return_curve:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            return auroc, fpr, tpr
            
        return auroc
        
    except ImportError:
        # Fallback implementation
        return _compute_auroc_manual(y_true, y_scores, return_curve)


def _compute_auroc_manual(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    return_curve: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """Manual AUROC computation without sklearn."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Sort by scores
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Compute TPR and FPR at each threshold
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5 if not return_curve else (0.5, None, None)
        
    tpr = np.cumsum(y_true_sorted == 1) / n_pos
    fpr = np.cumsum(y_true_sorted == 0) / n_neg
    
    # Add (0,0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Compute AUC using trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    
    if return_curve:
        return auroc, fpr, tpr
        
    return auroc


def compute_auprc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    return_curve: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """Compute Area Under Precision-Recall Curve.
    
    Important for imbalanced datasets (HAI: 11% positive).
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        return_curve: If True, also return precision and recall arrays
        
    Returns:
        AUPRC value, optionally (auprc, precision, recall)
    """
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
        
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class in y_true. AUPRC undefined.")
            baseline = np.mean(y_true)
            return baseline if not return_curve else (baseline, None, None)
            
        auprc = average_precision_score(y_true, y_scores)
        
        if return_curve:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            return auprc, precision, recall
            
        return auprc
        
    except ImportError:
        # Simplified fallback
        return np.mean(y_true)  # Baseline


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """Compute precision and recall."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute F1 score."""
    precision, recall = compute_precision_recall(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)


def compute_mcc(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute Matthews Correlation Coefficient.
    
    Good for imbalanced datasets - ranges from -1 to 1.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator


def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute specificity (true negative rate)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if tn + fp == 0:
        return 0.0
        
    return tn / (tn + fp)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1"
) -> float:
    """Find optimal classification threshold.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'youden')
        
    Returns:
        Optimal threshold value
    """
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_score = -np.inf
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        if metric == "f1":
            score = compute_f1(y_true, y_pred)
        elif metric == "accuracy":
            score = compute_accuracy(y_true, y_pred)
        elif metric == "youden":
            # Youden's J = TPR + TNR - 1
            _, recall = compute_precision_recall(y_true, y_pred)
            specificity = compute_specificity(y_true, y_pred)
            score = recall + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        if score > best_score:
            best_score = score
            best_threshold = thresh
            
    return best_threshold


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000
) -> BinaryClassificationMetrics:
    """Compute all classification metrics.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        threshold: Classification threshold (if None, optimizes)
        bootstrap_ci: Compute bootstrap confidence intervals
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        BinaryClassificationMetrics object
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Remove NaN
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_scores))
    y_true = y_true[valid_mask]
    y_scores = y_scores[valid_mask]
    
    if len(y_true) == 0:
        logger.warning("No valid samples for metric computation")
        return BinaryClassificationMetrics()
        
    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_scores)
        
    # Convert to predictions
    y_pred = (y_scores >= threshold).astype(int)
    
    # Compute metrics
    auroc = compute_auroc(y_true, y_scores)
    auprc = compute_auprc(y_true, y_scores)
    accuracy = compute_accuracy(y_true, y_pred)
    precision, recall = compute_precision_recall(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    mcc = compute_mcc(y_true, y_pred)
    specificity = compute_specificity(y_true, y_pred)
    
    # Bootstrap confidence intervals for AUROC
    auroc_ci = None
    if bootstrap_ci:
        auroc_ci = bootstrap_auroc_ci(y_true, y_scores, n_bootstrap)
        
    return BinaryClassificationMetrics(
        auroc=auroc,
        auprc=auprc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        specificity=specificity,
        threshold=threshold,
        n_samples=len(y_true),
        n_positive=int(np.sum(y_true == 1)),
        n_negative=int(np.sum(y_true == 0)),
        auroc_ci=auroc_ci
    )


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for AUROC.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        (lower, upper) confidence bounds
    """
    n = len(y_true)
    aurocs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, n, replace=True)
        
        # Check if both classes present
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        auroc = compute_auroc(y_true[indices], y_scores[indices])
        aurocs.append(auroc)
        
    if len(aurocs) < 100:
        logger.warning("Insufficient valid bootstrap samples")
        return (0.0, 1.0)
        
    alpha = 1 - confidence
    lower = np.percentile(aurocs, alpha/2 * 100)
    upper = np.percentile(aurocs, (1 - alpha/2) * 100)
    
    return (lower, upper)


def compare_to_baseline(
    metrics: BinaryClassificationMetrics,
    split_type: str,
    task: str = "binding"
) -> Dict[str, any]:
    """Compare metrics to paper baselines.
    
    Args:
        metrics: Computed metrics
        split_type: Split strategy ('lenient', 'ha_exclusive', 'mab_exclusive', 'mab_cluster')
        task: Task type ('binding' or 'hai')
        
    Returns:
        Comparison results
    """
    if task not in PAPER_BENCHMARKS:
        raise ValueError(f"Unknown task: {task}")
        
    if split_type not in PAPER_BENCHMARKS[task]:
        raise ValueError(f"Unknown split type: {split_type}")
        
    baseline = PAPER_BENCHMARKS[task][split_type]
    baseline_auroc = baseline['auroc']
    baseline_std = baseline['std']
    
    diff = metrics.auroc - baseline_auroc
    improvement_pct = (diff / baseline_auroc) * 100
    
    # Statistical significance (approximate)
    significant = abs(diff) > 2 * baseline_std
    
    result = {
        'your_auroc': metrics.auroc,
        'baseline_auroc': baseline_auroc,
        'baseline_std': baseline_std,
        'difference': diff,
        'improvement_pct': improvement_pct,
        'significant': significant,
        'status': 'BEAT' if diff > baseline_std else ('MATCH' if diff > -baseline_std else 'BELOW')
    }
    
    # Log result
    status_emoji = '🎉' if result['status'] == 'BEAT' else ('✓' if result['status'] == 'MATCH' else '⚠️')
    logger.info(f"{status_emoji} {split_type} {task}: {metrics.auroc:.4f} vs {baseline_auroc:.2f}±{baseline_std:.2f} [{result['status']}]")
    
    return result


def aggregate_cv_metrics(
    fold_metrics: List[BinaryClassificationMetrics]
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across CV folds.
    
    Args:
        fold_metrics: List of metrics from each fold
        
    Returns:
        Dictionary with mean and std for each metric
    """
    metric_names = ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'specificity']
    
    result = {}
    
    for name in metric_names:
        values = [getattr(m, name) for m in fold_metrics]
        result[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        
    return result


def format_cv_results(
    aggregated: Dict[str, Dict[str, float]],
    split_type: str = "lenient"
) -> str:
    """Format CV results as a summary string.
    
    Args:
        aggregated: Aggregated metrics from aggregate_cv_metrics
        split_type: Name of the split strategy
        
    Returns:
        Formatted summary string
    """
    lines = [
        f"\n{'='*60}",
        f"Cross-Validation Results: {split_type.upper()}",
        f"{'='*60}",
    ]
    
    for name, stats in aggregated.items():
        lines.append(f"  {name:12s}: {stats['mean']:.4f} ± {stats['std']:.4f} [{stats['min']:.4f} - {stats['max']:.4f}]")
        
    lines.append(f"{'='*60}\n")
    
    return '\n'.join(lines)
