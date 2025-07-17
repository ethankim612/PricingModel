import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def load_simulation_results(results_dir="binomial_monte_carlo_results"):
    """Load saved simulation results."""
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} folder not found.")
        return None
    
    results = {}
    
    # Load basic simulation data
    if os.path.exists(os.path.join(results_dir, 'simulation_data.npy')):
        basic_data = np.load(os.path.join(results_dir, 'simulation_data.npy'), allow_pickle=True).item()
        results['basic'] = basic_data
        print("✓ Basic simulation data loaded")
    
    # Load realistic parameter simulation data
    if os.path.exists(os.path.join(results_dir, 'realistic_simulation_data.npy')):
        realistic_data = np.load(os.path.join(results_dir, 'realistic_simulation_data.npy'), allow_pickle=True).item()
        results['realistic'] = realistic_data
        print("✓ Realistic parameter simulation data loaded")
    
    # Load cumulative sales data
    if os.path.exists(os.path.join(results_dir, 'cumulative_sales_data.npy')):
        cumulative_data = np.load(os.path.join(results_dir, 'cumulative_sales_data.npy'), allow_pickle=True).item()
        results['cumulative'] = cumulative_data
        print("✓ Cumulative sales data loaded")
    
    # Load results summary
    if os.path.exists(os.path.join(results_dir, 'results_summary.json')):
        with open(os.path.join(results_dir, 'results_summary.json'), 'r') as f:
            results['summary'] = json.load(f)
        print("✓ Results summary loaded")
    
    return results

def analyze_basic_simulation(data):
    """Analyze basic simulation results."""
    if 'basic' not in data:
        print("Basic simulation data not available.")
        return
    
    basic = data['basic']
    sales_paths = basic['sales_paths']
    closure_months = basic['closure_months']
    expected_sales = basic['expected_sales']
    survival_prob = basic['survival_prob']
    
    print("\n=== Basic Simulation Analysis ===")
    print(f"Closure probability (within 12 months): {np.mean(closure_months < 12) * 100:.1f}%")
    print(f"Average survival period: {closure_months.mean():.2f} months")
    print(f"Survival probability (12 months): {survival_prob[12] * 100:.1f}%")
    
    # Monthly sales statistics
    monthly_stats = []
    for month in range(sales_paths.shape[1]):
        monthly_sales = sales_paths[:, month]
        monthly_stats.append({
            'month': month,
            'mean_sales': np.mean(monthly_sales),
            'median_sales': np.median(monthly_sales),
            'std_sales': np.std(monthly_sales),
            'survival_prob': survival_prob[month],
            'percentile_10': np.percentile(monthly_sales, 10),
            'percentile_90': np.percentile(monthly_sales, 90)
        })
    
    monthly_df = pd.DataFrame(monthly_stats)
    print("\nMonthly Sales Statistics:")
    print(monthly_df.round(0))
    
    return monthly_df

def analyze_realistic_simulation(data):
    """Analyze realistic parameter simulation results."""
    if 'realistic' not in data:
        print("Realistic parameter simulation data not available.")
        return
    
    realistic = data['realistic']
    sales_paths = realistic['sales_paths']
    closure_months = realistic['closure_months']
    expected_sales = realistic['expected_sales']
    survival_prob = realistic['survival_prob']
    optimal_month = realistic['optimal_month']
    optimal_value = realistic['optimal_value']
    
    print("\n=== Realistic Parameter Simulation Analysis ===")
    print(f"Closure probability (within 12 months): {np.mean(closure_months < 12) * 100:.1f}%")
    print(f"Average survival period: {closure_months.mean():.2f} months")
    print(f"Survival probability (12 months): {survival_prob[12] * 100:.1f}%")
    print(f"Optimal exit month: {optimal_month} months")
    print(f"Optimal exit cumulative sales: {optimal_value:,.0f} KRW")
    
    # Parameter information
    if 'parameters' in realistic:
        params = realistic['parameters']
        print(f"\nSimulation Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    return realistic

def compare_simulations(data):
    """Compare two simulations."""
    if 'basic' not in data or 'realistic' not in data:
        print("Both simulation datasets are required.")
        return
    
    basic = data['basic']
    realistic = data['realistic']
    
    comparison = {
        'metric': ['Closure Probability (12 months)', 'Average Survival Period', 'Survival Probability (12 months)', 'Optimal Exit Month', 'Optimal Exit Cumulative Sales'],
        'basic': [
            np.mean(basic['closure_months'] < 12) * 100,
            basic['closure_months'].mean(),
            basic['survival_prob'][12] * 100,
            data['cumulative']['optimal_month'] if 'cumulative' in data else 'N/A',
            data['cumulative']['optimal_value'] if 'cumulative' in data else 'N/A'
        ],
        'realistic': [
            np.mean(realistic['closure_months'] < 12) * 100,
            realistic['closure_months'].mean(),
            realistic['survival_prob'][12] * 100,
            realistic['optimal_month'],
            realistic['optimal_value']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison)
    print("\n=== Simulation Comparison ===")
    print(comparison_df)
    
    return comparison_df

def plot_comparison(data):
    """Display comparison graphs of two simulation results."""
    if 'basic' not in data or 'realistic' not in data:
        print("Both simulation datasets are required.")
        return
    
    basic = data['basic']
    realistic = data['realistic']
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Survival probability comparison
    months = range(len(basic['survival_prob']))
    axs[0, 0].plot(months, basic['survival_prob'], 'b-', label='Basic Simulation', linewidth=2)
    axs[0, 0].plot(months, realistic['survival_prob'], 'r-', label='Realistic Parameters', linewidth=2)
    axs[0, 0].set_title('Survival Probability Comparison')
    axs[0, 0].set_xlabel('Month')
    axs[0, 0].set_ylabel('Survival Probability')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Expected sales comparison
    axs[0, 1].plot(months, basic['expected_sales'], 'b-', label='Basic Simulation', linewidth=2)
    axs[0, 1].plot(months, realistic['expected_sales'], 'r-', label='Realistic Parameters', linewidth=2)
    axs[0, 1].set_title('Expected Sales Comparison')
    axs[0, 1].set_xlabel('Month')
    axs[0, 1].set_ylabel('Sales (KRW)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Closure month distribution comparison
    axs[1, 0].hist(basic['closure_months'], bins=range(1, 14), alpha=0.7, label='Basic Simulation', color='blue')
    axs[1, 0].hist(realistic['closure_months'], bins=range(1, 14), alpha=0.7, label='Realistic Parameters', color='red')
    axs[1, 0].set_title('Closure Month Distribution Comparison')
    axs[1, 0].set_xlabel('Closure Month')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].legend()
    
    # 4. Cumulative sales comparison
    if 'cumulative' in data:
        cumulative = data['cumulative']
        axs[1, 1].plot(months, cumulative['expected_cumulative'], 'b-', label='Basic Simulation', linewidth=2)
        axs[1, 1].plot(months, realistic['expected_cumulative'], 'r-', label='Realistic Parameters', linewidth=2)
        axs[1, 1].set_title('Cumulative Sales Comparison')
        axs[1, 1].set_xlabel('Month')
        axs[1, 1].set_ylabel('Cumulative Sales (KRW)')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_detailed_report(data, output_file="binomial_analysis_report.txt"):
    """Generate detailed analysis report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Binomial Monte Carlo Simulation Analysis Report ===\n\n")
        
        if 'summary' in data:
            summary = data['summary']
            f.write("1. Basic Simulation Results:\n")
            basic = summary['basic_simulation']
            f.write(f"   - Closure probability (12 months): {basic['closure_probability_12_months']:.1f}%\n")
            f.write(f"   - Average survival period: {basic['average_survival_period']:.2f} months\n")
            f.write(f"   - Optimal exit month: {basic['optimal_exit_month']} months\n")
            f.write(f"   - Optimal exit cumulative sales: {basic['optimal_exit_value']:,.0f} KRW\n\n")
            
            f.write("2. Realistic Parameter Simulation Results:\n")
            realistic = summary['realistic_simulation']
            f.write(f"   - Closure probability (12 months): {realistic['closure_probability_12_months']:.1f}%\n")
            f.write(f"   - Average survival period: {realistic['average_survival_period']:.2f} months\n")
            f.write(f"   - Optimal exit month: {realistic['optimal_exit_month']} months\n")
            f.write(f"   - Optimal exit cumulative sales: {realistic['optimal_exit_value']:,.0f} KRW\n\n")
        
        if 'basic' in data and 'realistic' in data:
            f.write("3. Simulation Comparison:\n")
            basic_surv = data['basic']['survival_prob'][12] * 100
            realistic_surv = data['realistic']['survival_prob'][12] * 100
            f.write(f"   - Basic simulation 12-month survival probability: {basic_surv:.1f}%\n")
            f.write(f"   - Realistic parameters 12-month survival probability: {realistic_surv:.1f}%\n")
            f.write(f"   - Survival probability difference: {realistic_surv - basic_surv:.1f}%p\n\n")
    
    print(f"✓ Detailed report saved to {output_file}")

def main():
    """Main function"""
    print("Binomial Monte Carlo Results Analyzer")
    print("=" * 50)
    
    # Load results
    data = load_simulation_results()
    
    if data is None:
        print("Cannot load results.")
        return
    
    # Run analysis
    print("\n" + "="*50)
    print("Starting analysis...")
    
    # Basic simulation analysis
    basic_stats = analyze_basic_simulation(data)
    
    # Realistic parameter analysis
    realistic_stats = analyze_realistic_simulation(data)
    
    # Simulation comparison
    comparison = compare_simulations(data)
    
    # Comparison graphs
    plot_comparison(data)
    
    # Generate detailed report
    create_detailed_report(data)
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("Generated files:")
    print("- binomial_analysis_report.txt: Detailed analysis report")

if __name__ == "__main__":
    main() 