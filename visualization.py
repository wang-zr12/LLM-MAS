import pandas as pd

from eval import bucket_count, metrics
import matplotlib.pyplot as plt
import numpy as np

models = ['blank','sa','cot','sa_r1','mas','mas_drop_sp','mas_drop_cri']


def plot_multi_model_comparison(models, metrics):
    # Define colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Determine bucket boundaries for x-axis labels
        if metric:
            bucket_labels = ['0-195', '196-362', '363-529', '530-700', '700-900', '900+']
        else:
            bucket_labels = ['0-195', '196-362', '363-529', '530-695', '696-862',
                             '863-1029', '1030-1362', '1363+']

        for i, model in enumerate(models):
            excel_file = f'middle_results/{model}_count.xlsx'
            # Call bucket_count with draw=False to avoid individual plots
            normalized_scores, counts = bucket_count(excel_file, metric, draw=False)

            # Print bucket counts for debugging
            print(f"Model: {model}, Metric: {metric}, Bucket Counts: {counts}")

            # Plot line for this model
            color = colors[i % len(colors)]
            plt.plot(range(1, len(normalized_scores) + 1), normalized_scores,
                     marker='o', linestyle='-', color=color, label=model, linewidth=2, markersize=6)

        # Customize the plot
        title = metric if metric else "Total"

        plt.xlabel("Total Length Bucket", fontsize=12)
        plt.ylabel("Average Score", fontsize=12)
        plt.xticks(range(1, len(bucket_labels) + 1), bucket_labels, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot
        img_name = metric if metric else 'all'
        plt.savefig(f'{img_name}_multi_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
def plot_avg_score(excel_file):
    x_pos = range(len(models))
    results_df=pd.read_excel(excel_file)
    results_df.plot(kind='bar', figsize=(12, 6))
    plt.title("Model Performance by Metric Group")
    plt.ylabel("Weighted Average Score")
    plt.xticks(
        [x + 0.2 for x in x_pos],  # Center the labels
        models,
        rotation=45,
        ha='right',
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig("model_metric_comparison.png")
    plt.show()
def main():
    # Call the multi-model plotting function
    plot_multi_model_comparison(models, ['Exposure'])

if __name__ == "__main__":
    plot_avg_score('model_metric_analysis.xlsx')