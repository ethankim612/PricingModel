import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Font settings for matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class CafeSurvivalSimulation:
    def __init__(self):
        # Actual data (July-December 2024)
        self.actual_data = {
            7: 2911800,   # July (peak season)
            8: 3560000,   # August (peak season)
            9: 1708800,   # September (off season)
            10: 1634700,  # October (off season)
            11: 1204500,  # November (off season)
            12: 928500    # December (off season)
        }
        
        # Seasonal analysis
        self.peak_months = [6, 7, 8]  # Peak season (summer)
        self.off_months = [1, 2, 3, 4, 5, 9, 10, 11, 12]  # Off season
        
        self.analyze_seasonal_patterns()
        
    def analyze_seasonal_patterns(self):
        """Analyze seasonal patterns"""
        revenues = list(self.actual_data.values())
        
        # Peak season data (July-August)
        peak_data = [self.actual_data[7], self.actual_data[8]]
        # Off season data (September-December)
        off_data = [self.actual_data[9], self.actual_data[10], 
                   self.actual_data[11], self.actual_data[12]]
        
        # Peak season statistics
        self.peak_mean = np.mean(peak_data)
        self.peak_std = np.std(peak_data, ddof=1) if len(peak_data) > 1 else self.peak_mean * 0.15
        
        # Off season statistics
        self.off_mean = np.mean(off_data)
        self.off_std = np.std(off_data, ddof=1)
        
        # Calculate monthly return rates
        monthly_returns = []
        for i in range(1, len(revenues)):
            monthly_returns.append((revenues[i] - revenues[i-1]) / revenues[i-1])
        
        self.return_mean = np.mean(monthly_returns)
        self.return_std = np.std(monthly_returns, ddof=1)
        
        # Trend analysis (declining trend)
        months = list(range(len(revenues)))
        slope, intercept, r_value, p_value, std_err = stats.linregress(months, revenues)
        self.trend_slope = slope  # Monthly decline amount
        
        print("=== Seasonal Analysis Results ===")
        print(f"Peak season average: {self.peak_mean:,.0f} KRW (std: {self.peak_std:,.0f} KRW)")
        print(f"Off season average: {self.off_mean:,.0f} KRW (std: {self.off_std:,.0f} KRW)")
        print(f"Off season to peak season ratio: {self.off_mean/self.peak_mean:.1%}")
        print(f"Monthly average change rate: {self.return_mean:.2%}")
        print(f"Monthly change rate std: {self.return_std:.2%}")
        print(f"Declining trend: {self.trend_slope:,.0f} KRW decrease per month")
        
    def get_seasonal_params(self, month):
        """Return seasonal parameters for each month"""
        if month in self.peak_months:
            return self.peak_mean, self.peak_std
        else:
            return self.off_mean, self.off_std
            
    def simulate_revenue(self, month, base_revenue, trend_factor=1.0):
        """Monthly revenue simulation"""
        # Apply seasonality
        seasonal_mean, seasonal_std = self.get_seasonal_params(month)
        
        # Apply seasonality to base revenue
        if month in self.peak_months:
            seasonal_multiplier = np.random.normal(1.8, 0.3)  # Peak season 1.8x (±30%)
        else:
            seasonal_multiplier = np.random.normal(1.0, 0.2)  # Off season baseline (±20%)
        
        # Apply trend (decline over time)
        trend_adjusted_revenue = base_revenue * trend_factor
        
        # Apply monthly volatility
        monthly_variation = np.random.normal(1 + self.return_mean, self.return_std)
        
        # Calculate final revenue
        simulated_revenue = trend_adjusted_revenue * seasonal_multiplier * monthly_variation
        
        # Minimum revenue limit (prevent complete zero)
        return max(simulated_revenue, 100000)  # Minimum 100,000 KRW
        
    def run_simulation(self, initial_cash=10000000, fixed_costs=2000000, 
                      variable_cost_ratio=0.4, max_months=60, num_simulations=10000):
        """Run Monte Carlo simulation"""
        print(f"\n=== Simulation Settings ===")
        print(f"Initial capital: {initial_cash:,} KRW")
        print(f"Monthly fixed costs: {fixed_costs:,} KRW")
        print(f"Variable cost ratio: {variable_cost_ratio:.1%}")
        print(f"Maximum simulation period: {max_months} months")
        print(f"Number of simulations: {num_simulations:,}")
        
        results = []
        monthly_cash_flows = []
        
        for sim in range(num_simulations):
            cash = initial_cash
            survived_months = 0
            base_revenue = self.off_mean  # Base revenue (off season average)
            cash_history = [cash]
            
            for month_idx in range(max_months):
                # Calculate current month (January=1, December=12)
                current_month = ((month_idx + 1) % 12) + 1
                
                # Trend factor (decline over time)
                trend_factor = 1 + (self.trend_slope / self.off_mean) * month_idx
                trend_factor = max(trend_factor, 0.3)  # Minimum 30% decline
                
                # Revenue simulation
                revenue = self.simulate_revenue(current_month, base_revenue, trend_factor)
                
                # Cost calculation
                variable_costs = revenue * variable_cost_ratio
                total_costs = fixed_costs + variable_costs
                
                # Monthly profit
                monthly_profit = revenue - total_costs
                cash += monthly_profit
                
                cash_history.append(cash)
                
                # Bankruptcy condition
                if cash <= 0:
                    break
                    
                survived_months += 1
                
                # Store cash flow for first simulation
                if sim == 0:
                    monthly_cash_flows.append({
                        'month': month_idx + 1,
                        'calendar_month': current_month,
                        'revenue': revenue,
                        'costs': total_costs,
                        'profit': monthly_profit,
                        'cash': cash,
                        'trend_factor': trend_factor
                    })
            
            results.append({
                'survived_months': survived_months,
                'final_cash': cash,
                'went_bankrupt': cash <= 0
            })
        
        return results, monthly_cash_flows
    
    def analyze_results(self, results):
        """Analyze results"""
        df = pd.DataFrame(results)
        
        survival_months = df['survived_months']
        bankruptcy_rate = df['went_bankrupt'].mean()
        
        print(f"\n=== Simulation Results ===")
        print(f"Bankruptcy probability: {bankruptcy_rate:.1%}")
        print(f"Average survival period: {survival_months.mean():.1f} months")
        print(f"Median survival period: {survival_months.median():.1f} months")
        print(f"Standard deviation: {survival_months.std():.1f} months")
        
        # Survival probability analysis by period
        survival_probs = []
        periods = [6, 12, 18, 24, 36, 48]
        
        for period in periods:
            prob = (survival_months >= period).mean()
            survival_probs.append(prob)
            print(f"{period}-month survival probability: {prob:.1%}")
        
        return {
            'survival_months': survival_months,
            'bankruptcy_rate': bankruptcy_rate,
            'survival_probs': survival_probs,
            'periods': periods
        }
    
    def plot_results(self, results, monthly_cash_flows):
        """Visualize results"""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Survival period histogram
        axes[0, 0].hist(df['survived_months'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Survival Period Distribution')
        axes[0, 0].set_xlabel('Months')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df['survived_months'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["survived_months"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Survival probability curve
        max_months = int(df['survived_months'].max())
        survival_curve = []
        months_range = range(1, max_months + 1)
        
        for month in months_range:
            prob = (df['survived_months'] >= month).mean()
            survival_curve.append(prob)
        
        axes[0, 1].plot(months_range, survival_curve, linewidth=2)
        axes[0, 1].set_title('Survival Probability Curve')
        axes[0, 1].set_xlabel('Months')
        axes[0, 1].set_ylabel('Survival Probability')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cash flow (first simulation)
        if monthly_cash_flows:
            cash_df = pd.DataFrame(monthly_cash_flows)
            axes[1, 0].plot(cash_df['month'], cash_df['cash'], linewidth=2)
            axes[1, 0].set_title('Cash Flow Example (First Simulation)')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Cash (KRW)')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Monthly revenue pattern (first simulation)
        if monthly_cash_flows:
            axes[1, 1].bar(cash_df['month'], cash_df['revenue'], alpha=0.7)
            axes[1, 1].set_title('Monthly Revenue Pattern (First Simulation)')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Revenue (KRW)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional analysis graphs
        self.plot_scenario_analysis(results)
    
    def plot_scenario_analysis(self, results):
        """Scenario analysis graphs"""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Survival period category analysis
        bins = [0, 6, 12, 18, 24, 36, 48, 60, float('inf')]
        labels = ['0-6m', '6-12m', '12-18m', '18-24m', '24-36m', '36-48m', '48-60m', '60m+']
        
        df['survival_category'] = pd.cut(df['survived_months'], bins=bins, labels=labels)
        survival_counts = df['survival_category'].value_counts().sort_index()
        
        axes[0].bar(range(len(survival_counts)), survival_counts.values, alpha=0.7)
        axes[0].set_title('Survival Period Categories')
        axes[0].set_xlabel('Survival Period')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(len(survival_counts)))
        axes[0].set_xticklabels(survival_counts.index, rotation=45)
        
        # 2. Bankruptcy vs survival ratio
        bankruptcy_status = df['went_bankrupt'].value_counts()
        # Match data and labels count
        labels = ['Survived' if i == False else 'Bankrupt' for i in bankruptcy_status.index]
        axes[1].pie(bankruptcy_status.values, 
                   labels=labels, 
                   autopct='%1.1f%%',
                   startangle=90)
        axes[1].set_title('Bankruptcy vs Survival Rate')
        
        plt.tight_layout()
        plt.show()

# Execution example
def main():
    # Create simulation
    sim = CafeSurvivalSimulation()
    
    # Run simulation (multiple scenarios)
    scenarios = [
        {"name": "Current Situation", "initial_cash": 10000000, "fixed_costs": 2000000, "variable_cost_ratio": 0.4},
        {"name": "Pessimistic", "initial_cash": 5000000, "fixed_costs": 2500000, "variable_cost_ratio": 0.5},
        {"name": "Optimistic", "initial_cash": 20000000, "fixed_costs": 1500000, "variable_cost_ratio": 0.3},
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*50}") #dd
        
        results, cash_flows = sim.run_simulation(
            initial_cash=scenario['initial_cash'],
            fixed_costs=scenario['fixed_costs'],
            variable_cost_ratio=scenario['variable_cost_ratio'],
            max_months=60,
            num_simulations=10000
        )
        
        analysis = sim.analyze_results(results)
        
        # Only show graphs for first scenario
        if scenario['name'] == "Current Situation":
            sim.plot_results(results, cash_flows)

if __name__ == "__main__":
    main()