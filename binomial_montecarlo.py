import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

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

# Plotting all results in one figure
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# 1. Sample paths
for i in range(100):
    axs[0, 0].plot(range(forecast_months+1), sales_paths[i], alpha=0.15, color='blue')
axs[0, 0].plot(range(forecast_months+1), expected_sales, color='black', linewidth=2, label='Expected Sales (Mean)')
axs[0, 0].plot(range(forecast_months+1), median_sales, color='orange', linewidth=2, label='Median Sales')
axs[0, 0].axhline(close_threshold, color='red', linestyle='--', label='Closure Threshold')
axs[0, 0].set_title('Sample Sales Paths (100) with Mean/Median')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Sales (KRW)')
axs[0, 0].legend()

# 2. Closure month histogram
sns.histplot(closure_months, bins=np.arange(1, forecast_months+2)-0.5, kde=False, ax=axs[0, 1], color='purple')
axs[0, 1].set_title('Closure Month Distribution')
axs[0, 1].set_xlabel('Months Until Closure')
axs[0, 1].set_ylabel('Frequency')

# 3. Survival probability curve
axs[1, 0].plot(range(forecast_months+1), survival_prob, marker='o', color='green')
axs[1, 0].set_title('Monthly Survival Probability Curve')
axs[1, 0].set_xlabel('Month')
axs[1, 0].set_ylabel('Survival Probability')
axs[1, 0].set_ylim(0, 1.05)

# 4. Boxplot of sales by month (distribution insight)
sns.boxplot(data=sales_paths[:, 1:], ax=axs[1, 1], color='lightblue', showfliers=False)
axs[1, 1].set_title('Sales Distribution by Month (Boxplot)')
axs[1, 1].set_xlabel('Month')
axs[1, 1].set_ylabel('Sales (KRW)')

plt.tight_layout()
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
plt.plot(range(forecast_months+1), expected_cumulative, label='Expected Cumulative Sales (if exit now)')
plt.axvline(optimal_month, color='red', linestyle='--', label=f'Optimal Exit Month: {optimal_month}')
plt.scatter(optimal_month, optimal_value, color='red', zorder=5)
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
plt.tight_layout()
plt.show()

# 1. 현실적 파라미터
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

# 1. 현실적 파라미터로 시뮬레이션
result = run_simulation(**params_realistic)

print('==== Realistic Parameter Simulation ====')
print(f"Average sales: {mean_sales:,.0f} KRW")
print(f"Recent 3 months avg: {params_realistic['initial_sales']:,.0f} KRW")
print(f"Closure threshold: {params_realistic['close_threshold']:,.0f} KRW")
print(f"Up factor: {params_realistic['up_factor']}, Down factor: {params_realistic['down_factor']}, Up prob: {params_realistic['up_prob']}")
print(f"Closure probability within 12 months: {np.mean(result['closure_months'] < forecast_months) * 100:.1f}%")
print(f"Average survival period: {result['closure_months'].mean():.2f} months")
print(f"Optimal exit month: {result['optimal_month']} (expected cumulative sales: {result['optimal_value']:,.0f} KRW)")

# 그래프
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
for i in range(100):
    axs[0, 0].plot(range(forecast_months+1), result['sales_paths'][i], alpha=0.15, color='blue')
axs[0, 0].plot(range(forecast_months+1), result['expected_sales'], color='black', linewidth=2, label='Expected Sales (Mean)')
axs[0, 0].plot(range(forecast_months+1), result['median_sales'], color='orange', linewidth=2, label='Median Sales')
axs[0, 0].axhline(params_realistic['close_threshold'], color='red', linestyle='--', label='Closure Threshold')
axs[0, 0].set_title('Sample Sales Paths (100) with Mean/Median')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Sales (KRW)')
axs[0, 0].legend()

sns.histplot(result['closure_months'], bins=np.arange(1, forecast_months+2)-0.5, kde=False, ax=axs[0, 1], color='purple')
axs[0, 1].set_title('Closure Month Distribution')
axs[0, 1].set_xlabel('Months Until Closure')
axs[0, 1].set_ylabel('Frequency')

axs[1, 0].plot(range(forecast_months+1), result['survival_prob'], marker='o', color='green')
axs[1, 0].set_title('Monthly Survival Probability Curve')
axs[1, 0].set_xlabel('Month')
axs[1, 0].set_ylabel('Survival Probability')
axs[1, 0].set_ylim(0, 1.05)

sns.boxplot(data=result['sales_paths'][:, 1:], ax=axs[1, 1], color='lightblue', showfliers=False)
axs[1, 1].set_title('Sales Distribution by Month (Boxplot)')
axs[1, 1].set_xlabel('Month')
axs[1, 1].set_ylabel('Sales (KRW)')

plt.tight_layout()
plt.show()

# 누적 수익 & 최적 폐업 시점
plt.figure(figsize=(12,6))
plt.plot(range(forecast_months+1), result['expected_cumulative'], label='Expected Cumulative Sales (if exit now)')
plt.axvline(result['optimal_month'], color='red', linestyle='--', label=f'Optimal Exit Month: {result['optimal_month']}')
plt.scatter(result['optimal_month'], result['optimal_value'], color='red', zorder=5)
plt.annotate(f'Optimal Exit\\nMonth: {result['optimal_month']}\\nKRW: {int(result['optimal_value']):,}',
             xy=(result['optimal_month'], result['optimal_value']),
             xytext=(result['optimal_month']+0.5, result['optimal_value']*0.95),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', lw=1))
for i in range(10):
    plt.plot(range(forecast_months+1), result['cumulative_sales'][i], alpha=0.3, color='gray')
plt.title('Expected Cumulative Sales vs. Exit Timing')
plt.xlabel('Month')
plt.ylabel('Cumulative Sales (KRW)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. 여러 파라미터 조합 실험 (그리드 서치)
param_grid = {
    'close_threshold': [mean_sales * 0.5, mean_sales * 0.6, mean_sales * 0.7],
    'up_factor': [1.10, 1.15, 1.20],
    'down_factor': [0.90, 0.95, 1.00],
    'up_prob': [0.55, 0.60, 0.65]
}
import itertools
results = []
for close_thr, up_f, down_f, up_p in itertools.product(
        param_grid['close_threshold'],
        param_grid['up_factor'],
        param_grid['down_factor'],
        param_grid['up_prob']):
    res = run_simulation(close_thr, up_f, down_f, up_p, sales[-3:].mean())
    results.append({
        'close_threshold': close_thr,
        'up_factor': up_f,
        'down_factor': down_f,
        'up_prob': up_p,
        'closure_prob': np.mean(res['closure_months'] < forecast_months),
        'avg_survival': res['closure_months'].mean(),
        'optimal_month': res['optimal_month'],
        'optimal_value': res['optimal_value']
    })

df = pd.DataFrame(results)
print('\n==== Parameter Grid Search Results (Top 10 by optimal_value) ====')
print(df.sort_values('optimal_value', ascending=False).head(10))

# 최적 조합의 기대 누적 수익 곡선 그래프
best = df.sort_values('optimal_value', ascending=False).iloc[0]
best_result = run_simulation(
    best['close_threshold'], best['up_factor'], best['down_factor'], best['up_prob'], sales[-3:].mean()
)
plt.figure(figsize=(12,6))
plt.plot(range(forecast_months+1), best_result['expected_cumulative'], label='Best Expected Cumulative Sales')
plt.axvline(best_result['optimal_month'], color='red', linestyle='--', label=f'Optimal Exit Month: {best_result['optimal_month']}')
plt.scatter(best_result['optimal_month'], best_result['optimal_value'], color='red', zorder=5)
plt.title('Best Parameter Set: Expected Cumulative Sales vs. Exit Timing')
plt.xlabel('Month')
plt.ylabel('Cumulative Sales (KRW)')
plt.legend()
plt.tight_layout()
plt.show()

# 1. 입주율 및 소비파워 반영
households = 1000
expected_spending_per_household = 30000  # 월평균 카페 이용액
market_share = 0.2  # 카페 점유율
max_market_sales = households * expected_spending_per_household * market_share

# 2. 입주율 곡선 (예: 7월~12월, 이후는 100%)
move_in_rate = [0, 0, 0, 0.4, 0.8, 1.0] + [1.0] * 6  # 12개월 예시

# 3. up/down factor, up_prob 동적 변화
up_factor_list = [1.07, 1.07, 1.07, 1.10, 1.15, 1.18] + [1.18] * 6
down_factor_list = [0.93, 0.93, 0.93, 0.97, 1.00, 1.02] + [1.02] * 6
up_prob_list = [0.53, 0.53, 0.53, 0.58, 0.62, 0.65] + [0.65] * 6

# 4. 시뮬레이션
num_simulations = 100_000
forecast_months = 12
initial_sales = sales[-3:].mean()
close_threshold = mean_sales * 0.6

rand_matrix = np.random.rand(num_simulations, forecast_months)
sales_paths = np.ones((num_simulations, forecast_months+1)) * initial_sales

for t in range(1, forecast_months+1):
    up_factor = up_factor_list[t-1]
    down_factor = down_factor_list[t-1]
    up_prob = up_prob_list[t-1]
    factors = np.where(rand_matrix[:, t-1] < up_prob, up_factor, down_factor)
    sales_paths[:, t] = sales_paths[:, t-1] * factors
    # 시장 최대 매출 한계 적용
    sales_paths[:, t] = np.minimum(sales_paths[:, t], max_market_sales * move_in_rate[t-1])

# 폐업 처리
closure_months = np.full(num_simulations, forecast_months)
for t in range(1, forecast_months+1):
    just_closed = (sales_paths[:, t] < close_threshold) & (closure_months == forecast_months)
    closure_months[just_closed] = t
    sales_paths[just_closed, t:] = 0

# 이후 누적 수익, 최적 폐업 시점 등 기존 분석 동일하게 진행

# --- Economic Parameter Grid Search & Visualization ---
households = 1000
market_share_list = [0.1, 0.2, 0.3]
expected_spending_list = [20000, 30000, 40000]
close_threshold_ratio_list = [0.5, 0.6, 0.7]
up_factor_list = [1.10, 1.15, 1.20]
down_factor_list = [0.90, 0.95, 1.00]
up_prob_list = [0.55, 0.60, 0.65]

forecast_months = 12
num_simulations = 1_000_000  # Large-scale simulation

results = []

def run_simulation_grid(initial_sales, close_threshold, up_factor, down_factor, up_prob, max_market_sales, move_in_rate):
    rand_matrix = np.random.rand(num_simulations, forecast_months)
    sales_paths = np.ones((num_simulations, forecast_months+1)) * initial_sales
    for t in range(1, forecast_months+1):
        factors = np.where(rand_matrix[:, t-1] < up_prob, up_factor, down_factor)
        sales_paths[:, t] = sales_paths[:, t-1] * factors
        sales_paths[:, t] = np.minimum(sales_paths[:, t], max_market_sales * move_in_rate[t-1])
    closure_months = np.full(num_simulations, forecast_months)
    for t in range(1, forecast_months+1):
        just_closed = (sales_paths[:, t] < close_threshold) & (closure_months == forecast_months)
        closure_months[just_closed] = t
        sales_paths[just_closed, t:] = 0
    cumulative_sales = np.cumsum(sales_paths, axis=1)
    expected_cumulative = cumulative_sales.mean(axis=0)
    optimal_month = np.argmax(expected_cumulative)
    optimal_value = expected_cumulative[optimal_month]
    return optimal_month, optimal_value, np.mean(closure_months < forecast_months)

# Move-in rate curve (example)
move_in_rate = [0, 0, 0, 0.4, 0.8, 1.0] + [1.0] * 6

for market_share, expected_spending, close_thr_ratio, up_f, down_f, up_p in itertools.product(
        market_share_list, expected_spending_list, close_threshold_ratio_list,
        up_factor_list, down_factor_list, up_prob_list):
    max_market_sales = households * expected_spending * market_share
    initial_sales = max_market_sales * move_in_rate[0] if move_in_rate[0] > 0 else max_market_sales * 0.1
    close_threshold = max_market_sales * close_thr_ratio
    optimal_month, optimal_value, closure_prob = run_simulation_grid(
        initial_sales, close_threshold, up_f, down_f, up_p, max_market_sales, move_in_rate
    )
    results.append({
        'market_share': market_share,
        'expected_spending': expected_spending,
        'close_threshold_ratio': close_thr_ratio,
        'up_factor': up_f,
        'down_factor': down_f,
        'up_prob': up_p,
        'optimal_month': optimal_month,
        'optimal_value': optimal_value,
        'closure_prob': closure_prob
    })

df = pd.DataFrame(results)

# 1. Scatter: Optimal Exit Month vs. Expected Cumulative Sales
plt.figure(figsize=(10,6))
scatter = plt.scatter(df['optimal_month'], df['optimal_value'], c=df['closure_prob'], cmap='viridis', alpha=0.7)
plt.title('Optimal Exit Month vs. Expected Cumulative Sales')
plt.xlabel('Optimal Exit Month')
plt.ylabel('Expected Cumulative Sales (KRW)')
cbar = plt.colorbar(scatter)
cbar.set_label('Closure Probability')
plt.tight_layout()
plt.show()

# 2. Heatmap: Expected Cumulative Sales by Up Factor & Up Probability
pivot = df.pivot_table(index='up_factor', columns='up_prob', values='optimal_value', aggfunc='max')
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Expected Cumulative Sales (Heatmap)\nby Up Factor & Up Probability')
plt.xlabel('Up Probability')
plt.ylabel('Up Factor')
plt.tight_layout()
plt.show()

# 3. Line plot: Expected Cumulative Sales by Market Share & Spending
plt.figure(figsize=(10,6))
for ms in market_share_list:
    subset = df[df['market_share'] == ms]
    plt.plot(subset['expected_spending'], subset['optimal_value'], marker='o', label=f'Market Share {ms*100:.0f}%')
plt.title('Expected Cumulative Sales by Market Share & Spending')
plt.xlabel('Expected Spending per Household (KRW)')
plt.ylabel('Expected Cumulative Sales (KRW)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Save simulation summary to CSV ---
# Calculate standard deviation and 95% confidence interval for expected sales
std_sales = sales_paths.std(axis=0)
ci_upper = expected_sales + 1.96 * std_sales / np.sqrt(num_simulations)
ci_lower = expected_sales - 1.96 * std_sales / np.sqrt(num_simulations)

# Prepare result dictionary
result_dict = {
    'close_threshold': [close_threshold],
    'up_factor': [up_factor],
    'down_factor': [down_factor],
    'up_prob': [up_prob],
    'initial_sales': [initial_sales],
    'optimal_exit_month': [optimal_month],
    'expected_cumulative_sales': [optimal_value],
    'avg_survival_months': [closure_months.mean()],
    'closure_probability': [np.mean(closure_months < forecast_months)]
}
for m in range(forecast_months+1):
    result_dict[f'expected_sales_month_{m}'] = [expected_sales[m]]
    result_dict[f'ci_lower_month_{m}'] = [ci_lower[m]]
    result_dict[f'ci_upper_month_{m}'] = [ci_upper[m]]

result_df = pd.DataFrame(result_dict)
result_df.to_csv('simulation_summary.csv', index=False)
print('Saved summary results to simulation_summary.csv')

# --- Plot expected sales with 95% confidence interval ---
plt.figure(figsize=(12,6))
plt.plot(range(forecast_months+1), expected_sales, label='Expected Sales')
plt.fill_between(range(forecast_months+1), ci_lower, ci_upper, color='gray', alpha=0.3, label='95% CI')
plt.axhline(close_threshold, color='red', linestyle='--', label='Closure Threshold')
plt.title('Expected Sales with 95% Confidence Interval')
plt.xlabel('Month')
plt.ylabel('Sales (KRW)')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 추가 시각화 및 통계 자료
# =========================

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 월별 폐업 확률 (bar chart)
plt.figure(figsize=(10,4))
closure_counts = np.bincount(closure_months, minlength=forecast_months+1)
closure_rate = closure_counts[1:] / num_simulations
plt.bar(range(1, forecast_months+1), closure_rate, color='crimson', alpha=0.7)
plt.title('월별 폐업 확률')
plt.xlabel('폐업 발생 월')
plt.ylabel('폐업 확률')
plt.tight_layout()
plt.show()

# 2. 카플란-마이어 생존 곡선
try:
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(closure_months, event_observed=(closure_months < forecast_months))
    plt.figure(figsize=(10,4))
    kmf.plot_survival_function()
    plt.title('카플란-마이어 생존 곡선')
    plt.xlabel('개월')
    plt.ylabel('생존 확률')
    plt.tight_layout()
    plt.show()
except ImportError:
    print('lifelines 패키지가 설치되어 있지 않아 카플란-마이어 곡선을 그릴 수 없습니다.')

# 3. 누적 매출 분포 (12, 24, 36개월)
for m in [12, 24, 36]:
    if m <= forecast_months:
        plt.figure(figsize=(8,4))
        plt.hist(np.cumsum(sales_paths, axis=1)[:, m], bins=50, color='navy', alpha=0.7)
        plt.title(f'{m}개월 시점 누적 매출 분포')
        plt.xlabel('누적 매출 (KRW)')
        plt.ylabel('빈도')
        plt.tight_layout()
        plt.show()

# 4. 폐업 시점별 누적 매출 boxplot
plt.figure(figsize=(12,6))
data = [np.cumsum(sales_paths, axis=1)[closure_months == m, m] for m in range(1, forecast_months+1) if np.sum(closure_months == m) > 0]
plt.boxplot(data, positions=range(1, len(data)+1), showfliers=False)
plt.title('폐업 시점별 누적 매출 분포')
plt.xlabel('폐업 월')
plt.ylabel('누적 매출 (KRW)')
plt.tight_layout()
plt.show()

# 5. 상위/하위 10% 경로 시각화
final_sales = sales_paths[:, -1]
top_idx = np.argsort(final_sales)[-10:]
bottom_idx = np.argsort(final_sales)[:10]
plt.figure(figsize=(12,6))
for idx in top_idx:
    plt.plot(range(forecast_months+1), sales_paths[idx], color='blue', alpha=0.7)
for idx in bottom_idx:
    plt.plot(range(forecast_months+1), sales_paths[idx], color='red', alpha=0.7)
plt.title('상위 10% (파랑) / 하위 10% (빨강) 매출 경로')
plt.xlabel('개월')
plt.ylabel('매출 (KRW)')
plt.tight_layout()
plt.show()

# 6. 파라미터-결과 상관관계 히트맵 (df가 있을 때)
try:
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('파라미터-결과 상관관계 히트맵')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print('파라미터-결과 상관관계 히트맵을 그릴 수 없습니다:', e)

# 7. 생존/폐업 경로만 따로 시각화
plt.figure(figsize=(12,6))
for i in np.where(closure_months == forecast_months)[0][:50]:
    plt.plot(range(forecast_months+1), sales_paths[i], color='green', alpha=0.2)
for i in np.where(closure_months < forecast_months)[0][:50]:
    plt.plot(range(forecast_months+1), sales_paths[i], color='red', alpha=0.2)
plt.title('생존(녹색)/폐업(빨강) 경로 샘플')
plt.xlabel('개월')
plt.ylabel('매출 (KRW)')
plt.tight_layout()
plt.show()

# 8. 월별 평균/중간값/상위10%/하위10% 매출 라인
mean_sales_line = sales_paths.mean(axis=0)
median_sales_line = np.median(sales_paths, axis=0)
top10 = np.percentile(sales_paths, 90, axis=0)
bottom10 = np.percentile(sales_paths, 10, axis=0)
plt.figure(figsize=(12,6))
plt.plot(mean_sales_line, label='평균')
plt.plot(median_sales_line, label='중간값')
plt.plot(top10, label='상위 10%')
plt.plot(bottom10, label='하위 10%')
plt.title('월별 매출 통계')
plt.xlabel('개월')
plt.ylabel('매출 (KRW)')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 폐업까지 걸린 기간 분포
plt.figure(figsize=(8,4))
plt.hist(closure_months, bins=forecast_months, color='purple', alpha=0.7)
plt.title('폐업까지 걸린 기간 분포')
plt.xlabel('개월')
plt.ylabel('빈도')
plt.tight_layout()
plt.show()

# =========================
# 가격 예측 및 평가 기능 (simul.py 데이터 활용)
# =========================

print("\n=== 가격 예측 및 평가 분석 ===")

# simul.py의 실제 데이터 활용
actual_data = {
    7: 2911800,   # 7월 (성수기)
    8: 3560000,   # 8월 (성수기)
    9: 1708800,   # 9월 (비성수기)
    10: 1634700,  # 10월 (비성수기)
    11: 1204500,  # 11월 (비성수기)
    12: 928500    # 12월 (비성수기)
}

# 계절성 분석
peak_months = [6, 7, 8]  # 성수기 (여름)
off_months = [1, 2, 3, 4, 5, 9, 10, 11, 12]  # 비성수기

# 성수기/비성수기 통계
peak_data = [actual_data[7], actual_data[8]]
off_data = [actual_data[9], actual_data[10], actual_data[11], actual_data[12]]

peak_mean = np.mean(peak_data)
peak_std = np.std(peak_data, ddof=1) if len(peak_data) > 1 else peak_mean * 0.15
off_mean = np.mean(off_data)
off_std = np.std(off_data, ddof=1)

# 트렌드 분석
revenues = list(actual_data.values())
months = list(range(len(revenues)))
slope, intercept, r_value, p_value, std_err = stats.linregress(months, revenues)
trend_slope = slope

print(f"성수기 평균: {peak_mean:,.0f}원")
print(f"비성수기 평균: {off_mean:,.0f}원")
print(f"하락 트렌드: 월 {trend_slope:,.0f}원씩 감소")

# 가격 예측 함수
def predict_price(month, base_revenue, trend_factor=1.0):
    """월별 가격 예측"""
    # 계절성 적용
    if month in peak_months:
        seasonal_mean, seasonal_std = peak_mean, peak_std
        seasonal_multiplier = np.random.normal(1.8, 0.3)  # 성수기 1.8배
    else:
        seasonal_mean, seasonal_std = off_mean, off_std
        seasonal_multiplier = np.random.normal(1.0, 0.2)  # 비성수기 기본
    
    # 트렌드 적용
    trend_adjusted_revenue = base_revenue * trend_factor
    
    # 월별 변동성
    monthly_variation = np.random.normal(1, 0.15)  # 15% 변동성
    
    # 최종 예측
    predicted_price = trend_adjusted_revenue * seasonal_multiplier * monthly_variation
    return max(predicted_price, 100000)

# 가격 예측 시뮬레이션
price_predictions = []
price_actuals = []
price_errors = []

for month in range(1, 13):  # 12개월
    # 실제 데이터가 있는 경우
    if month in actual_data:
        actual = actual_data[month]
        price_actuals.append(actual)
        
        # 예측값 계산 (여러 시뮬레이션)
        predictions = []
        for _ in range(1000):
            trend_factor = 1 + (trend_slope / off_mean) * (month - 1)
            trend_factor = max(trend_factor, 0.3)
            pred = predict_price(month, off_mean, trend_factor)
            predictions.append(pred)
        
        avg_prediction = np.mean(predictions)
        price_predictions.append(avg_prediction)
        price_errors.append(actual - avg_prediction)
        
        print(f"{month}월: 실제 {actual:,.0f}원, 예측 {avg_prediction:,.0f}원, 오차 {actual-avg_prediction:,.0f}원")

# 예측 성능 평가
if len(price_actuals) > 0:
    actuals = np.array(price_actuals)
    predictions = np.array(price_predictions)
    errors = np.array(price_errors)
    
    # 평가 지표
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actuals)) * 100
    
    print(f"\n=== 예측 성능 평가 ===")
    print(f"RMSE: {rmse:,.0f}원")
    print(f"MAE: {mae:,.0f}원")
    print(f"MAPE: {mape:.1f}%")

# 가격 예측 시각화
plt.figure(figsize=(12, 6))
months_with_data = list(actual_data.keys())
actual_values = list(actual_data.values())

plt.plot(months_with_data, actual_values, 'o-', label='실제 매출', linewidth=2, markersize=8)
plt.plot(months_with_data, price_predictions, 's--', label='예측 매출', linewidth=2, markersize=8)

plt.fill_between(months_with_data, 
                [p - np.std([predict_price(m, off_mean, 1 + (trend_slope / off_mean) * (m - 1)) for _ in range(100)]) for m, p in zip(months_with_data, price_predictions)],
                [p + np.std([predict_price(m, off_mean, 1 + (trend_slope / off_mean) * (m - 1)) for _ in range(100)]) for m, p in zip(months_with_data, price_predictions)],
                alpha=0.3, label='예측 구간')

plt.title('월별 매출 예측 vs 실제')
plt.xlabel('월')
plt.ylabel('매출 (KRW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 오차 분석
plt.figure(figsize=(10, 6))
plt.bar(months_with_data, price_errors, alpha=0.7, color='red')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.title('월별 예측 오차')
plt.xlabel('월')
plt.ylabel('오차 (실제 - 예측)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 가격 예측 결과를 CSV로 저장
price_analysis = []
for i, month in enumerate(months_with_data):
    price_analysis.append({
        'month': month,
        'actual_revenue': actual_values[i],
        'predicted_revenue': price_predictions[i],
        'error': price_errors[i],
        'error_percentage': (price_errors[i] / actual_values[i]) * 100
    })

price_analysis_df = pd.DataFrame(price_analysis)
price_analysis_df.to_csv('price_prediction_analysis.csv', index=False)
print("✓ price_prediction_analysis.csv 저장 완료")

# 예측 성능 요약
performance_summary = {
    'metric': ['RMSE', 'MAE', 'MAPE (%)', 'R²'],
    'value': [rmse, mae, mape, r_value**2],
    'description': ['Root Mean Square Error', 'Mean Absolute Error', 'Mean Absolute Percentage Error', 'R-squared']
}

performance_df = pd.DataFrame(performance_summary)
performance_df.to_csv('prediction_performance.csv', index=False)
print("✓ prediction_performance.csv 저장 완료")

# =========================
# CSV 파일 저장 기능
# =========================

print("\n=== CSV 파일 저장 중 ===")

# 1. 시뮬레이션 요약 결과 (기존)
result_df.to_csv('long_term_simulation_summary.csv', index=False)
print("✓ long_term_simulation_summary.csv 저장 완료")

# 2. 월별 통계 데이터
monthly_stats = []
for month in range(forecast_months + 1):
    monthly_stats.append({
        'month': month,
        'expected_sales': expected_sales[month],
        'median_sales': np.median(sales_paths[:, month]),
        'std_sales': np.std(sales_paths[:, month]),
        'survival_probability': survival_prob[month],
        'closure_probability': 1 - survival_prob[month],
        'min_sales': np.min(sales_paths[:, month]),
        'max_sales': np.max(sales_paths[:, month]),
        'percentile_10': np.percentile(sales_paths[:, month], 10),
        'percentile_25': np.percentile(sales_paths[:, month], 25),
        'percentile_75': np.percentile(sales_paths[:, month], 75),
        'percentile_90': np.percentile(sales_paths[:, month], 90)
    })

monthly_stats_df = pd.DataFrame(monthly_stats)
monthly_stats_df.to_csv('monthly_statistics.csv', index=False)
print("✓ monthly_statistics.csv 저장 완료")

# 3. 폐업 분석 데이터
closure_analysis = []
cumulative_closure_prob = 0
for month in range(1, forecast_months + 1):
    monthly_closure_count = np.sum(closure_months == month)
    monthly_closure_prob = monthly_closure_count / num_simulations
    cumulative_closure_prob += monthly_closure_prob
    
    closure_analysis.append({
        'month': month,
        'closure_count': monthly_closure_count,
        'monthly_closure_probability': monthly_closure_prob,
        'cumulative_closure_probability': cumulative_closure_prob,
        'survival_probability': 1 - cumulative_closure_prob,
        'avg_cumulative_sales_at_closure': np.mean(cumulative_sales[closure_months == month, month]) if monthly_closure_count > 0 else 0
    })

closure_analysis_df = pd.DataFrame(closure_analysis)
closure_analysis_df.to_csv('closure_analysis.csv', index=False)
print("✓ closure_analysis.csv 저장 완료")

# 4. 누적 매출 분석 데이터
cumulative_analysis = []
for month in range(forecast_months + 1):
    cumulative_analysis.append({
        'month': month,
        'expected_cumulative_sales': expected_cumulative[month],
        'median_cumulative_sales': np.median(cumulative_sales[:, month]),
        'std_cumulative_sales': np.std(cumulative_sales[:, month]),
        'min_cumulative_sales': np.min(cumulative_sales[:, month]),
        'max_cumulative_sales': np.max(cumulative_sales[:, month]),
        'percentile_10_cumulative': np.percentile(cumulative_sales[:, month], 10),
        'percentile_25_cumulative': np.percentile(cumulative_sales[:, month], 25),
        'percentile_75_cumulative': np.percentile(cumulative_sales[:, month], 75),
        'percentile_90_cumulative': np.percentile(cumulative_sales[:, month], 90)
    })

cumulative_analysis_df = pd.DataFrame(cumulative_analysis)
cumulative_analysis_df.to_csv('cumulative_sales_analysis.csv', index=False)
print("✓ cumulative_sales_analysis.csv 저장 완료")

# 5. 시뮬레이션 파라미터 요약
parameter_summary = {
    'parameter': [
        'forecast_months', 'num_simulations', 'initial_sales', 'close_threshold',
        'up_factor', 'down_factor', 'up_prob', 'households', 'expected_spending_per_household',
        'market_share', 'max_market_sales'
    ],
    'value': [
        forecast_months, num_simulations, initial_sales, close_threshold,
        up_factor, down_factor, up_prob, households, expected_spending_per_household,
        market_share, max_market_sales
    ],
    'description': [
        '예측 기간 (개월)', '시뮬레이션 횟수', '초기 매출', '폐업 임계값',
        '상승 팩터', '하락 팩터', '상승 확률', '가구 수', '가구당 월평균 소비액',
        '시장 점유율', '최대 시장 매출'
    ]
}

parameter_summary_df = pd.DataFrame(parameter_summary)
parameter_summary_df.to_csv('parameter_summary.csv', index=False)
print("✓ parameter_summary.csv 저장 완료")

# 6. 주요 결과 요약 (간단한 버전)
key_results = {
    'metric': [
        '5년 내 폐업 확률 (%)',
        '평균 생존 기간 (개월)',
        '최적 폐업 시점 (개월)',
        '최적 폐업 시점 누적 매출 (KRW)',
        '12개월 생존 확률 (%)',
        '24개월 생존 확률 (%)',
        '36개월 생존 확률 (%)',
        '48개월 생존 확률 (%)',
        '60개월 생존 확률 (%)'
    ],
    'value': [
        np.mean(closure_months < forecast_months) * 100,
        closure_months.mean(),
        optimal_month,
        optimal_value,
        survival_prob[12] * 100,
        survival_prob[24] * 100,
        survival_prob[36] * 100,
        survival_prob[48] * 100,
        survival_prob[60] * 100
    ]
}

key_results_df = pd.DataFrame(key_results)
key_results_df.to_csv('key_results_summary.csv', index=False)
print("✓ key_results_summary.csv 저장 완료")

print("\n=== 모든 CSV 파일 저장 완료 ===")
print("생성된 파일들:")
print("- long_term_simulation_summary.csv: 상세 시뮬레이션 결과")
print("- monthly_statistics.csv: 월별 통계")
print("- closure_analysis.csv: 폐업 분석")
print("- cumulative_sales_analysis.csv: 누적 매출 분석")
print("- parameter_summary.csv: 시뮬레이션 파라미터")
print("- key_results_summary.csv: 주요 결과 요약")