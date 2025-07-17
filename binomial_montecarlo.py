import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy.stats as stats
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Create results directory
results_dir = "binomial_monte_carlo_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Data
sales = np.array([5847600, 7239000, 3417600, 3269400, 2409000, 1857000])
months = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"]

mean_sales = sales.mean()
close_threshold = mean_sales * 0.8

# Use more moderate, realistic up/down factors
up_factor = 1.07  # 7% up
down_factor = 0.93  # 7% down

# Use a constant up probability for simplicity and realism
up_prob = 0.53

initial_sales = sales[-1]
forecast_months = 12
num_simulations = 100_000

print("==== Simulation Parameters ====")
print(f"Average sales: {mean_sales:,.0f} KRW")
print(f"Closure threshold (80%): {close_threshold:,.0f} KRW")
print(f"Up factor (u): {up_factor:.3f}")
print(f"Down factor (d): {down_factor:.3f}")
print(f"Up probability: {up_prob:.2f}")
print(f"Number of simulations: {num_simulations}")
print(f"Forecast period: {forecast_months} months")

# Vectorized simulation
rand_matrix = np.random.rand(num_simulations, forecast_months)
up_mask = rand_matrix < up_prob
factors = np.where(up_mask, up_factor, down_factor)
sales_paths = np.ones((num_simulations, forecast_months+1)) * initial_sales

for t in range(1, forecast_months+1):
    sales_paths[:, t] = sales_paths[:, t-1] * factors[:, t-1]

# Apply closure threshold: after closure, set to 0
closure_months = np.full(num_simulations, forecast_months)
for t in range(1, forecast_months+1):
    just_closed = (sales_paths[:, t] < close_threshold) & (closure_months == forecast_months)
    closure_months[just_closed] = t
    # Set all future sales to 0 for closed paths
    sales_paths[just_closed, t:] = 0

expected_sales = sales_paths.mean(axis=0)
median_sales = np.median(sales_paths, axis=0)
survival_prob = (sales_paths > 0).mean(axis=0)

print("\n==== Simulation Results ====")
print(f"Closure probability within 12 months: {np.mean(closure_months < forecast_months) * 100:.1f}%")
print(f"Average survival period: {closure_months.mean():.2f} months")
print(f"First closure month (1=after 1 month): {closure_months.min()} ~ {closure_months.max()}")

# Save simulation data
simulation_data = {
    'sales_paths': sales_paths,
    'closure_months': closure_months,
    'expected_sales': expected_sales,
    'median_sales': median_sales,
    'survival_prob': survival_prob,
    'parameters': {
        'initial_sales': initial_sales,
        'close_threshold': close_threshold,
        'up_factor': up_factor,
        'down_factor': down_factor,
        'up_prob': up_prob,
        'forecast_months': forecast_months,
        'num_simulations': num_simulations
    }
}

# Save simulation data to numpy file
np.save(os.path.join(results_dir, 'simulation_data.npy'), simulation_data)
print(f"✓ Simulation data saved to {results_dir}/simulation_data.npy")

# 1. Sample paths plot
plt.figure(figsize=(12, 8))
for i in range(100):
    plt.plot(range(forecast_months+1), sales_paths[i], alpha=0.15, color='blue')
plt.plot(range(forecast_months+1), expected_sales, color='black', linewidth=2, label='Expected Sales (Mean)')
plt.plot(range(forecast_months+1), median_sales, color='orange', linewidth=2, label='Median Sales')
plt.axhline(close_threshold, color='red', linestyle='--', label='Closure Threshold')
plt.title('Sample Sales Paths (100) with Mean/Median')
plt.xlabel('Month')
plt.ylabel('Sales (KRW)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'sample_sales_paths.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Closure month histogram
plt.figure(figsize=(10, 6))
sns.histplot(closure_months, bins=np.arange(1, forecast_months+2)-0.5, kde=False, color='purple')
plt.title('Closure Month Distribution')
plt.xlabel('Months Until Closure')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'closure_month_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Survival probability curve
plt.figure(figsize=(10, 6))
plt.plot(range(forecast_months+1), survival_prob, marker='o', color='green', linewidth=2, markersize=6)
plt.title('Monthly Survival Probability Curve')
plt.xlabel('Month')
plt.ylabel('Survival Probability')
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'survival_probability_curve.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Boxplot of sales by month
plt.figure(figsize=(12, 6))
sns.boxplot(data=sales_paths[:, 1:], color='lightblue', showfliers=False)
plt.title('Sales Distribution by Month (Boxplot)')
plt.xlabel('Month')
plt.ylabel('Sales (KRW)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'sales_distribution_boxplot.png'), dpi=300, bbox_inches='tight')
plt.show()

# Cumulative sales (profit) analysis for optimal exit timing
cumulative_sales = np.cumsum(sales_paths, axis=1)
expected_cumulative = cumulative_sales.mean(axis=0)
optimal_month = np.argmax(expected_cumulative)
optimal_value = expected_cumulative[optimal_month]

print("\n==== Optimal Exit Analysis ====")
print(f"Best expected cumulative sales: {optimal_value:,.0f} KRW")
print(f"Optimal month to exit: {optimal_month} (0=now, 1=after 1 month, ...)")
print(f"Expected cumulative sales if you exit at each month: {expected_cumulative.astype(int)}")

# Plot: Expected cumulative sales vs. exit timing
plt.figure(figsize=(12,6))
plt.plot(range(forecast_months+1), expected_cumulative, label='Expected Cumulative Sales (if exit now)', linewidth=2)
plt.axvline(optimal_month, color='red', linestyle='--', label=f'Optimal Exit Month: {optimal_month}')
plt.scatter(optimal_month, optimal_value, color='red', zorder=5, s=100)
plt.annotate(f'Optimal Exit\nMonth: {optimal_month}\nKRW: {int(optimal_value):,}',
             xy=(optimal_month, optimal_value),
             xytext=(optimal_month+0.5, optimal_value*0.95),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', lw=1))
for i in range(10):
    plt.plot(range(forecast_months+1), cumulative_sales[i], alpha=0.3, color='gray')
plt.title('Expected Cumulative Sales vs. Exit Timing')
plt.xlabel('Month')
plt.ylabel('Cumulative Sales (KRW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'optimal_exit_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save cumulative sales data
cumulative_data = {
    'cumulative_sales': cumulative_sales,
    'expected_cumulative': expected_cumulative,
    'optimal_month': optimal_month,
    'optimal_value': optimal_value
}
np.save(os.path.join(results_dir, 'cumulative_sales_data.npy'), cumulative_data)
print(f"✓ Cumulative sales data saved to {results_dir}/cumulative_sales_data.npy")

# 1. Realistic parameters
params_realistic = {
    'close_threshold': mean_sales * 0.6,
    'up_factor': 1.15,
    'down_factor': 0.95,
    'up_prob': 0.60,
    'initial_sales': sales[-3:].mean()
}

def run_simulation(close_threshold, up_factor, down_factor, up_prob, initial_sales):
    rand_matrix = np.random.rand(num_simulations, forecast_months)
    up_mask = rand_matrix < up_prob
    factors = np.where(up_mask, up_factor, down_factor)
    sales_paths = np.ones((num_simulations, forecast_months+1)) * initial_sales

    for t in range(1, forecast_months+1):
        sales_paths[:, t] = sales_paths[:, t-1] * factors[:, t-1]

    closure_months = np.full(num_simulations, forecast_months)
    for t in range(1, forecast_months+1):
        just_closed = (sales_paths[:, t] < close_threshold) & (closure_months == forecast_months)
        closure_months[just_closed] = t
        sales_paths[just_closed, t:] = 0

    cumulative_sales = np.cumsum(sales_paths, axis=1)
    expected_cumulative = cumulative_sales.mean(axis=0)
    optimal_month = np.argmax(expected_cumulative)
    optimal_value = expected_cumulative[optimal_month]
    expected_sales = sales_paths.mean(axis=0)
    median_sales = np.median(sales_paths, axis=0)
    survival_prob = (sales_paths > 0).mean(axis=0)
    return {
        'sales_paths': sales_paths,
        'closure_months': closure_months,
        'cumulative_sales': cumulative_sales,
        'expected_cumulative': expected_cumulative,
        'optimal_month': optimal_month,
        'optimal_value': optimal_value,
        'expected_sales': expected_sales,
        'median_sales': median_sales,
        'survival_prob': survival_prob
    }

# 1. Run simulation with realistic parameters
result = run_simulation(**params_realistic)

print('==== Realistic Parameter Simulation ====')
print(f"Average sales: {mean_sales:,.0f} KRW")
print(f"Recent 3 months avg: {params_realistic['initial_sales']:,.0f} KRW")
print(f"Closure threshold: {params_realistic['close_threshold']:,.0f} KRW")
print(f"Up factor: {params_realistic['up_factor']}, Down factor: {params_realistic['down_factor']}, Up prob: {params_realistic['up_prob']}")
print(f"Closure probability within 12 months: {np.mean(result['closure_months'] < forecast_months) * 100:.1f}%")
print(f"Average survival period: {result['closure_months'].mean():.2f} months")
print(f"Optimal exit month: {result['optimal_month']} (expected cumulative sales: {result['optimal_value']:,.0f} KRW)")

# Save realistic simulation data
realistic_data = {
    'sales_paths': result['sales_paths'],
    'closure_months': result['closure_months'],
    'cumulative_sales': result['cumulative_sales'],
    'expected_cumulative': result['expected_cumulative'],
    'optimal_month': result['optimal_month'],
    'optimal_value': result['optimal_value'],
    'expected_sales': result['expected_sales'],
    'median_sales': result['median_sales'],
    'survival_prob': result['survival_prob'],
    'parameters': params_realistic
}
np.save(os.path.join(results_dir, 'realistic_simulation_data.npy'), realistic_data)
print(f"✓ Realistic simulation data saved to {results_dir}/realistic_simulation_data.npy")

# Graphs
# 1. Sample paths for realistic parameters
plt.figure(figsize=(12, 8))
for i in range(100):
    plt.plot(range(forecast_months+1), result['sales_paths'][i], alpha=0.15, color='blue')
plt.plot(range(forecast_months+1), result['expected_sales'], color='black', linewidth=2, label='Expected Sales (Mean)')
plt.plot(range(forecast_months+1), result['median_sales'], color='orange', linewidth=2, label='Median Sales')
plt.axhline(params_realistic['close_threshold'], color='red', linestyle='--', label='Closure Threshold')
plt.title('Sample Sales Paths (100) with Mean/Median - Realistic Parameters')
plt.xlabel('Month')
plt.ylabel('Sales (KRW)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'realistic_sample_sales_paths.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Closure month histogram for realistic parameters
plt.figure(figsize=(10, 6))
sns.histplot(result['closure_months'], bins=np.arange(1, forecast_months+2)-0.5, kde=False, color='purple')
plt.title('Closure Month Distribution - Realistic Parameters')
plt.xlabel('Months Until Closure')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'realistic_closure_month_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Survival probability curve for realistic parameters
plt.figure(figsize=(10, 6))
plt.plot(range(forecast_months+1), result['survival_prob'], marker='o', color='green', linewidth=2, markersize=6)
plt.title('Monthly Survival Probability Curve - Realistic Parameters')
plt.xlabel('Month')
plt.ylabel('Survival Probability')
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'realistic_survival_probability_curve.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Optimal exit analysis for realistic parameters
plt.figure(figsize=(12,6))
plt.plot(range(forecast_months+1), result['expected_cumulative'], label='Expected Cumulative Sales (if exit now)', linewidth=2)
plt.axvline(result['optimal_month'], color='red', linestyle='--', label=f'Optimal Exit Month: {result["optimal_month"]}')
plt.scatter(result['optimal_month'], result['optimal_value'], color='red', zorder=5, s=100)
plt.annotate(f'Optimal Exit\nMonth: {result["optimal_month"]}\nKRW: {int(result["optimal_value"]):,}',
             xy=(result['optimal_month'], result['optimal_value']),
             xytext=(result['optimal_month']+0.5, result['optimal_value']*0.95),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', lw=1))
for i in range(10):
    plt.plot(range(forecast_months+1), result['cumulative_sales'][i], alpha=0.3, color='gray')
plt.title('Expected Cumulative Sales vs. Exit Timing - Realistic Parameters')
plt.xlabel('Month')
plt.ylabel('Cumulative Sales (KRW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'realistic_optimal_exit_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save all results summary
results_summary = {
    'basic_simulation': {
        'closure_probability_12_months': np.mean(closure_months < forecast_months) * 100,
        'average_survival_period': closure_months.mean(),
        'optimal_exit_month': optimal_month,
        'optimal_exit_value': optimal_value,
        'survival_probabilities': survival_prob.tolist()
    },
    'realistic_simulation': {
        'closure_probability_12_months': np.mean(result['closure_months'] < forecast_months) * 100,
        'average_survival_period': result['closure_months'].mean(),
        'optimal_exit_month': result['optimal_month'],
        'optimal_exit_value': result['optimal_value'],
        'survival_probabilities': result['survival_prob'].tolist()
    },
    'parameters': {
        'basic': {
            'initial_sales': initial_sales,
            'close_threshold': close_threshold,
            'up_factor': up_factor,
            'down_factor': down_factor,
            'up_prob': up_prob
        },
        'realistic': params_realistic
    }
}

import json
with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
print(f"✓ Results summary saved to {results_dir}/results_summary.json")

print(f"\n=== All results saved to {results_dir} folder ===")
print("Saved files:")
print("- simulation_data.npy: Basic simulation data")
print("- realistic_simulation_data.npy: Realistic parameter simulation data")
print("- cumulative_sales_data.npy: Cumulative sales data")
print("- results_summary.json: Results summary")
print("- *.png: Various graphs")