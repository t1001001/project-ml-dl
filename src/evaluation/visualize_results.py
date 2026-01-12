import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_all(results):
    models = [r['name'] for r in results]
    classes = results[0]['class_names']
    
    accuracy = {r['name']: r['accuracy'] for r in results}
    f1_scores = {r['name']: r['f1_scores'] for r in results}
    precision_scores = {r['name']: r['precision_scores'] for r in results}
    recall_scores = {r['name']: r['recall_scores'] for r in results}
    f1_macro = {r['name']: r['f1_macro'] for r in results}
    f1_weighted = {r['name']: r['f1_weighted'] for r in results}
    inference_time_ms = {r['name']: r['inference_time_ms'] for r in results}
    confusion_matrices = {r['name']: r['confusion_matrix'] for r in results}
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    
    sci_colors = ['#d62728', '#2ca02c', '#1f77b4']
    hatches = ['', '///', '...']
    
    # Accuracy
    fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
    ax1.set_facecolor('white')
    x = np.arange(len(models))
    
    for i, model in enumerate(models):
        ax1.bar(x[i], accuracy[model], width=0.6, color=sci_colors[i], 
                edgecolor='black', linewidth=1, hatch=hatches[i], alpha=0.85)
        ax1.text(x[i], accuracy[model] + 0.02, f'{accuracy[model]:.0%}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison: Overall Classification Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlim(-0.5, 2.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax1.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/model_comparison_accuracy.png', dpi=300, bbox_inches='tight')
    plt.savefig('src/evaluation/model_comparison_accuracy.svg', bbox_inches='tight')
    print("Saved: model_comparison_accuracy.png/svg")
    
    # F1-Score per Class
    fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor='white')
    ax2.set_facecolor('white')
    x = np.arange(len(classes))
    width = 0.25
    
    for i, model in enumerate(models):
        offset = (i - 1) * width
        ax2.bar(x + offset, f1_scores[model], width * 0.85, 
                color=sci_colors[i], edgecolor='black', linewidth=0.8,
                label=model, hatch=hatches[i], alpha=0.85, zorder=2)
    
    ax2.axhline(y=0.9, color='#333333', linestyle='--', linewidth=1.5, zorder=3)
    ax2.annotate('Threshold = 0.9', xy=(len(classes) - 0.3, 0.9), xytext=(len(classes) - 0.3, 0.82),
                 fontsize=10, ha='right', va='top', color='#333333', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#333333', lw=1))
    
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Animal Class', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class F1-Score Comparison Across Models', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in classes], rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(-0.5, len(classes) - 0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc='lower right', frameon=True, framealpha=1, edgecolor='black', fontsize=10, fancybox=False)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/model_comparison_f1_classes.png', dpi=300, bbox_inches='tight')
    plt.savefig('src/evaluation/model_comparison_f1_classes.svg', bbox_inches='tight')
    print("Saved: model_comparison_f1_classes.png/svg")
    
    # Inference Time
    fig3, ax3 = plt.subplots(figsize=(8, 6), facecolor='white')
    ax3.set_facecolor('white')
    x = np.arange(len(models))
    
    for i, model in enumerate(models):
        ax3.bar(x[i], inference_time_ms[model], width=0.6, 
                color=sci_colors[i], edgecolor='black', linewidth=1, hatch=hatches[i], alpha=0.85)
        ax3.text(x[i], inference_time_ms[model] + 0.1, f'{inference_time_ms[model]:.2f} ms', 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax3.set_ylabel('Inference Time (ms per image)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax3.set_title('Model Comparison: Inference Time per Image', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=11)
    ax3.set_ylim(0, max(inference_time_ms.values()) * 1.25)
    ax3.set_xlim(-0.5, 2.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax3.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/model_comparison_inference_time.png', dpi=300, bbox_inches='tight')
    plt.savefig('src/evaluation/model_comparison_inference_time.svg', bbox_inches='tight')
    print("Saved: model_comparison_inference_time.png/svg")
    
    # Precision per Class
    fig4, ax4 = plt.subplots(figsize=(12, 6), facecolor='white')
    ax4.set_facecolor('white')
    x = np.arange(len(classes))
    width = 0.25
    
    for i, model in enumerate(models):
        offset = (i - 1) * width
        ax4.bar(x + offset, precision_scores[model], width * 0.85, 
                color=sci_colors[i], edgecolor='black', linewidth=0.8,
                label=model, hatch=hatches[i], alpha=0.85, zorder=2)
    
    ax4.axhline(y=0.9, color='#333333', linestyle='--', linewidth=1.5, zorder=3)
    ax4.annotate('Threshold = 0.9', xy=(len(classes) - 0.3, 0.9), xytext=(len(classes) - 0.3, 0.82),
                 fontsize=10, ha='right', va='top', color='#333333', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#333333', lw=1))
    
    ax4.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Animal Class', fontsize=12, fontweight='bold')
    ax4.set_title('Per-Class Precision Comparison Across Models', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.capitalize() for c in classes], rotation=45, ha='right', fontsize=10)
    ax4.set_ylim(0, 1.05)
    ax4.set_xlim(-0.5, len(classes) - 0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax4.set_axisbelow(True)
    ax4.legend(loc='lower right', frameon=True, framealpha=1, edgecolor='black', fontsize=10, fancybox=False)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/model_comparison_precision_classes.png', dpi=300, bbox_inches='tight')
    plt.savefig('src/evaluation/model_comparison_precision_classes.svg', bbox_inches='tight')
    print("Saved: model_comparison_precision_classes.png/svg")
    
    # Recall per Class
    fig5, ax5 = plt.subplots(figsize=(12, 6), facecolor='white')
    ax5.set_facecolor('white')
    x = np.arange(len(classes))
    width = 0.25
    
    for i, model in enumerate(models):
        offset = (i - 1) * width
        ax5.bar(x + offset, recall_scores[model], width * 0.85, 
                color=sci_colors[i], edgecolor='black', linewidth=0.8,
                label=model, hatch=hatches[i], alpha=0.85, zorder=2)
    
    ax5.axhline(y=0.9, color='#333333', linestyle='--', linewidth=1.5, zorder=3)
    ax5.annotate('Threshold = 0.9', xy=(len(classes) - 0.3, 0.9), xytext=(len(classes) - 0.3, 0.82),
                 fontsize=10, ha='right', va='top', color='#333333', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#333333', lw=1))
    
    ax5.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Animal Class', fontsize=12, fontweight='bold')
    ax5.set_title('Per-Class Recall Comparison Across Models', fontsize=14, fontweight='bold', pad=15)
    ax5.set_xticks(x)
    ax5.set_xticklabels([c.capitalize() for c in classes], rotation=45, ha='right', fontsize=10)
    ax5.set_ylim(0, 1.05)
    ax5.set_xlim(-0.5, len(classes) - 0.5)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax5.set_axisbelow(True)
    ax5.legend(loc='lower right', frameon=True, framealpha=1, edgecolor='black', fontsize=10, fancybox=False)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/model_comparison_recall_classes.png', dpi=300, bbox_inches='tight')
    plt.savefig('src/evaluation/model_comparison_recall_classes.svg', bbox_inches='tight')
    print("Saved: model_comparison_recall_classes.png/svg")
    
    # Macro and Weighted F1 Comparison
    fig6, ax6 = plt.subplots(figsize=(10, 6), facecolor='white')
    ax6.set_facecolor('white')
    x = np.arange(len(models))
    width = 0.35
    
    for i, model in enumerate(models):
        ax6.bar(x[i] - width/2, f1_macro[model], width * 0.85, 
                color=sci_colors[i], edgecolor='black', linewidth=1, 
                hatch=hatches[i], alpha=0.85, label='Macro F1' if i == 0 else '')
        ax6.bar(x[i] + width/2, f1_weighted[model], width * 0.85, 
                color=sci_colors[i], edgecolor='black', linewidth=1,
                hatch='', alpha=0.5, label='Weighted F1' if i == 0 else '')
        ax6.text(x[i] - width/2, f1_macro[model] + 0.015, f'{f1_macro[model]:.2f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax6.text(x[i] + width/2, f1_weighted[model] + 0.015, f'{f1_weighted[model]:.2f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax6.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax6.set_title('Model Comparison: Macro vs Weighted F1-Score', fontsize=14, fontweight='bold', pad=15)
    ax6.set_xticks(x)
    ax6.set_xticklabels(models, fontsize=11)
    ax6.set_ylim(0, 1.1)
    ax6.set_xlim(-0.5, len(models) - 0.5)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax6.set_axisbelow(True)
    
    # Custom legend for Macro vs Weighted
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='gray', edgecolor='black', hatch='///', alpha=0.85, label='Macro F1'),
                       Patch(facecolor='gray', edgecolor='black', alpha=0.5, label='Weighted F1')]
    ax6.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=1, 
               edgecolor='black', fontsize=10, fancybox=False)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/model_comparison_f1_macro_weighted.png', dpi=300, bbox_inches='tight')
    plt.savefig('src/evaluation/model_comparison_f1_macro_weighted.svg', bbox_inches='tight')
    print("Saved: model_comparison_f1_macro_weighted.png/svg")
    
    # Confusion Matrices
    model_filenames = {'Custom CNN': 'custom', 'MobileNetV2': 'mobilenetv2', 'ResNet50': 'resnet50'}
    
    for model in models:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')
        
        sns.heatmap(confusion_matrices[model], annot=True, fmt='d', cmap='Blues',
                    xticklabels=[c.capitalize() for c in classes],
                    yticklabels=[c.capitalize() for c in classes],
                    ax=ax, cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='white')
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('True', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {model}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        plt.tight_layout()
        filename = model_filenames[model]
        plt.savefig(f'src/evaluation/{filename}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'src/evaluation/{filename}_confusion_matrix.svg', bbox_inches='tight')
        print(f"Saved: {filename}_confusion_matrix.png/svg")
    
    plt.show()
    print("\n✅ All visualizations saved to src/evaluation/")

if __name__ == "__main__":
    print("Run: python -m evaluation.evaluation")
