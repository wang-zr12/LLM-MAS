import pandas as pd
import numpy as np
from scipy import stats


def calculate_score_volatility(excel_file, metric=None):
    """
    Calculate volatility metrics for scores in a specific metric

    Args:
        excel_file: Path to the Excel file
        metric: Specific metric to analyze (None for all data)

    Returns:
        dict: Dictionary containing various volatility metrics
    """
    df = pd.read_excel(excel_file)

    # Filter by metric if specified
    if metric:
        df_filtered = df[df["metric_en"] == metric]
    else:
        df_filtered = df

    if len(df_filtered) == 0:
        return {
            'count': 0,
            'mean': 0,
            'std': 0,
            'cv': 0,
            'range': 0,
            'iqr': 0,
            'mad': 0,
            'skewness': 0,
            'kurtosis': 0
        }

    scores = df_filtered['score'].values

    # Basic statistics
    count = len(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)  # Sample standard deviation

    # Coefficient of Variation (CV) - normalized volatility
    cv = (std_score / mean_score) * 100 if mean_score != 0 else 0

    # Range
    score_range = np.max(scores) - np.min(scores)

    # Interquartile Range (IQR)
    q75, q25 = np.percentile(scores, [75, 25])
    iqr = q75 - q25

    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(scores - mean_score))

    # Skewness and Kurtosis
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)

    return {
        'count': count,
        'mean': mean_score,
        'std': std_score,
        'cv': cv,
        'range': score_range,
        'iqr': iqr,
        'mad': mad,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def analyze_model_volatility(models, metrics, output_file):
    """
    Analyze score volatility for all models and metrics

    Args:
        models: List of model names
        metrics: List of metrics to analyze
    """
    print("=" * 80)
    print("SCORE VOLATILITY ANALYSIS")
    print("=" * 80)

    # Store all results for Excel export
    all_results = []

    for metric in metrics:
        print(f"\n{'=' * 60}")
        print(f"METRIC: {metric if metric else 'ALL METRICS'}")
        print(f"{'=' * 60}")

        # Store results for comparison
        results = {}

        for model in models:
            excel_file = f'middle_results/{model}_count.xlsx'
            volatility_metrics = calculate_score_volatility(excel_file, metric)
            results[model] = volatility_metrics

            # Prepare data for Excel export
            row_data = {
                'metric': metric if metric else 'ALL',
                'model_name': model,
                'data_points': volatility_metrics['count'],
                'mean_score': volatility_metrics['mean'],
                'std_deviation': volatility_metrics['std'],
                'coefficient_of_variation': volatility_metrics['cv'],
                'score_range': volatility_metrics['range'],
                'interquartile_range': volatility_metrics['iqr'],
                'mean_absolute_deviation': volatility_metrics['mad'],
                'skewness': volatility_metrics['skewness'],
                'kurtosis': volatility_metrics['kurtosis']
            }

            # Add volatility level interpretation
            cv = volatility_metrics['cv']
            if cv < 10:
                volatility_level = "Very Low"
            elif cv < 20:
                volatility_level = "Low"
            elif cv < 30:
                volatility_level = "Moderate"
            elif cv < 50:
                volatility_level = "High"
            else:
                volatility_level = "Very High"

            row_data['volatility_level'] = volatility_level
            all_results.append(row_data)

            print(f"\nModel: {model.upper()}")
            print(f"  Data Points: {volatility_metrics['count']}")
            print(f"  Mean Score: {volatility_metrics['mean']:.4f}")
            print(f"  Standard Deviation: {volatility_metrics['std']:.4f}")
            print(f"  Coefficient of Variation: {volatility_metrics['cv']:.2f}%")
            print(f"  Score Range: {volatility_metrics['range']:.4f}")
            print(f"  Interquartile Range (IQR): {volatility_metrics['iqr']:.4f}")
            print(f"  Mean Absolute Deviation: {volatility_metrics['mad']:.4f}")
            print(f"  Skewness: {volatility_metrics['skewness']:.4f}")
            print(f"  Kurtosis: {volatility_metrics['kurtosis']:.4f}")
            print(f"  Volatility Level: {volatility_level}")

        # Compare models for this metric
        print(f"\n{'-' * 40}")
        print(f"COMPARISON FOR {metric if metric else 'ALL METRICS'}:")
        print(f"{'-' * 40}")

        # Sort by coefficient of variation
        sorted_models = sorted(results.items(), key=lambda x: x[1]['cv'])

        print("Models ranked by volatility (CV - Coefficient of Variation):")
        for i, (model, metrics_data) in enumerate(sorted_models, 1):
            cv = metrics_data['cv']
            if cv < 20:
                stability = "🟢 Stable"
            elif cv < 40:
                stability = "🟡 Moderate"
            else:
                stability = "🔴 Volatile"

            print(f"  {i}. {model}: CV = {cv:.2f}% {stability}")

        # Find most and least volatile
        most_volatile = max(results.items(), key=lambda x: x[1]['cv'])
        least_volatile = min(results.items(), key=lambda x: x[1]['cv'])

        print(f"\nMost Volatile Model: {most_volatile[0]} (CV: {most_volatile[1]['cv']:.2f}%)")
        print(f"Least Volatile Model: {least_volatile[0]} (CV: {least_volatile[1]['cv']:.2f}%)")

    # Export to Excel
    df_results = pd.DataFrame(all_results)

    # Reorder columns for better readability
    column_order = [
        'metric', 'model_name', 'data_points', 'mean_score', 'std_deviation',
        'coefficient_of_variation', 'score_range', 'interquartile_range',
        'mean_absolute_deviation', 'skewness', 'kurtosis', 'volatility_level'
    ]
    df_results = df_results[column_order]

    # Save to Excel

    df_results.to_excel(output_file, index=False)

    print(f"\n{'=' * 80}")
    print(f"RESULTS EXPORTED TO: {output_file}")
    print(f"Total records: {len(df_results)}")
    print(f"{'=' * 80}")

    return df_results


def detailed_volatility_summary(volatility_file='volatility_analysis_results.xlsx'):
    """
    Create a summary table of volatility metrics from the exported Excel file

    Args:
        volatility_file: Path to the volatility analysis results Excel file
    """
    try:
        # Read the volatility analysis results
        df = pd.read_excel(volatility_file)

        print(f"\n{'=' * 100}")
        print("VOLATILITY SUMMARY TABLE")
        print(f"{'=' * 100}")

        # Create header
        header = f"{'Metric':<15} {'Model':<15} {'Mean':<8} {'Std':<8} {'CV%':<8} {'Range':<8} {'Level':<20}"
        print(header)
        print("-" * len(header))

        # Group by metric for better display
        metrics = df['metric'].unique()

        for metric in metrics:
            metric_data = df[df['metric'] == metric]

            for i, (_, row) in enumerate(metric_data.iterrows()):
                metric_display = metric if i == 0 else ""

                summary_row = f"{metric_display:<15} {row['model_name']:<15} {row['mean_score']:<8.3f} {row['std_deviation']:<8.3f} {row['coefficient_of_variation']:<8.1f} {row['score_range']:<8.3f} {row['volatility_level']:<20}"
                print(summary_row)

            if metric != metrics[-1]:  # Add separator between metrics
                print("-" * len(header))

        # Additional summary statistics
        print(f"\n{'=' * 100}")
        print("OVERALL SUMMARY STATISTICS")
        print(f"{'=' * 100}")

        print(f"Total Models Analyzed: {df['model_name'].nunique()}")
        print(f"Total Metrics Analyzed: {df['metric'].nunique()}")
        print(f"Total Records: {len(df)}")

        print(f"\nAverage CV by Model:")
        model_avg_cv = df.groupby('model_name')['coefficient_of_variation'].mean().sort_values()
        for model, avg_cv in model_avg_cv.items():
            print(f"  {model}: {avg_cv:.2f}%")

        print(f"\nAverage CV by Metric:")
        metric_avg_cv = df.groupby('metric')['coefficient_of_variation'].mean().sort_values()
        for metric, avg_cv in metric_avg_cv.items():
            print(f"  {metric}: {avg_cv:.2f}%")

        print(f"\nVolatility Level Distribution:")
        volatility_dist = df['volatility_level'].value_counts()
        for level, count in volatility_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")

    except FileNotFoundError:
        print(f"Error: Could not find {volatility_file}")
        print("Please run analyze_model_volatility() first to generate the results file.")
    except Exception as e:
        print(f"Error reading volatility analysis file: {e}")

# Usage
models = ['sa', 'cot', 'sa_r1', 'mas', 'mas_drop_sp', 'mas_drop_cri']
from eval import  metrics
output_file = 'my_results/volatility_analysis_results.xlsx'
# Run the volatility analysis
analyze_model_volatility(models, metrics, output_file)

# Show summary table
detailed_volatility_summary(output_file)