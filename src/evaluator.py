"""Evaluation utilities — metrics, confusion matrices, feature importance plots."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
)

PLOTS_DIR = Path(__file__).parent.parent / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

DARK_BG  = '#0b1220'
CARD_BG  = '#161b22'
ACCENT   = '#ffb347'
TEXT_COL = '#e6edf3'


def _style():
    plt.rcParams.update({
        'figure.facecolor': DARK_BG, 'axes.facecolor': CARD_BG,
        'text.color': TEXT_COL, 'axes.labelcolor': TEXT_COL,
        'xtick.color': TEXT_COL, 'ytick.color': TEXT_COL,
        'axes.edgecolor': '#30363d', 'grid.color': '#21262d',
        'font.family': 'DejaVu Sans',
    })


def print_report(y_true, y_pred, label='Model'):
    print(f'\n{"─"*50}')
    print(f'  {label} Evaluation')
    print(f'{"─"*50}')
    print(classification_report(y_true, y_pred))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f'  Macro F1: {macro_f1:.4f}')
    return macro_f1


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', filename='cm.png'):
    _style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor='#30363d',
                annot_kws={'size': 13, 'weight': 'bold', 'color': DARK_BG})
    ax.set_title(title, color=ACCENT, fontsize=14, pad=14)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'  Saved → {path}')


def plot_feature_importance(importances, feature_names, top_n=20, title='Feature Importance', filename='fi.png'):
    _style()
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.YlOrBr(np.linspace(0.3, 0.9, top_n))
    ax.barh([feature_names[i] for i in idx], importances[idx], color=colors)
    ax.set_title(title, color=ACCENT, fontsize=14, pad=12)
    ax.set_xlabel('Importance', fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'  Saved → {path}')


def plot_roc_curves(y_true, y_proba, class_labels, title='ROC Curves', filename='roc.png'):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    _style()
    n_classes = len(class_labels)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
        ax.set_title(title, color=ACCENT, fontsize=14)
        ax.legend(facecolor=CARD_BG, edgecolor='#30363d')
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [ACCENT, '#58a6ff', '#3fb950']
        for i, (lbl, col) in enumerate(zip(class_labels, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, color=col, lw=2, label=f'{lbl} (AUC={auc(fpr,tpr):.2f})')
        ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
        ax.set_title(title, color=ACCENT, fontsize=14)
        ax.legend(facecolor=CARD_BG, edgecolor='#30363d')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'  Saved → {path}')
