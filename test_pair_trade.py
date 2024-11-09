# test_pairs_trading.py
import pandas as pd
import numpy as np
from pathlib import Path
from basic_coint import PairsDataProcessor, PairsTradingStrategy
from statsmodels.tsa.stattools import coint
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns

def test_with_sample_data():
    """
    Further optimized test suite for pairs trading strategy
    """
    # Configuration
    DATA_FOLDER = Path('/Users/mouyasushi/k_data/永豐')
    OUTPUT_FOLDER = Path('./test_output')
    START_DATE = '2023-01-01'
    END_DATE = '2024-01-14'
    
    try:
        # 初始化處理器
        processor = PairsDataProcessor(DATA_FOLDER)
        all_stocks_daily = processor.combine_stock_data(START_DATE, END_DATE)
        
        # 增強版股票篩選
        def enhanced_filter_stocks(data):
            # 計算各種指標
            returns = data.pct_change()
            volatility = returns.std()
            avg_price = data.mean()
            daily_changes = abs(returns).mean()
            
            # 過濾條件
            valid_stocks = []
            for col in data.columns:
                price_series = data[col].dropna()
                if len(price_series) < len(data) * 0.9:  # 至少90%的數據完整性
                    continue
                    
                if (10 <= avg_price[col] <= 1000 and  # 合理價格區間
                    0.005 <= daily_changes[col] <= 0.03 and  # 適當的日均波動
                    volatility[col] <= 0.03):  # 控制總體波動率
                    valid_stocks.append(col)
            
            return valid_stocks
        
        # 優化配對選擇策略
        def enhanced_pair_selection(data, valid_stocks, max_pairs=20):
            pairs = []
            returns = data[valid_stocks].pct_change()
            
            # 計算相關性和波動率
            correlations = returns.corr()
            volatilities = returns.std()
            
            for i, stock1 in enumerate(valid_stocks):
                for stock2 in valid_stocks[i+1:]:
                    corr = correlations.loc[stock1, stock2]
                    if corr > 0.8:  # 高相關性要求
                        try:
                            # 計算價格比率的穩定性
                            price_ratio = data[stock1] / data[stock2]
                            ratio_std = price_ratio.std() / price_ratio.mean()
                            
                            # 協整合測試
                            _, p_value, _ = coint(data[stock1], data[stock2])
                            
                            if p_value < 0.05 and ratio_std < 0.1:  # 添加價格比率穩定性條件
                                # 計算配對分數
                                vol_diff = abs(volatilities[stock1] - volatilities[stock2])
                                pair_score = corr * (1 - vol_diff) * (1 - ratio_std)
                                pairs.append((stock1, stock2, corr, p_value, pair_score))
                                
                        except:
                            continue
            
            # 按綜合分數排序
            pairs.sort(key=lambda x: x[4], reverse=True)
            return pairs[:max_pairs]
        
        # 優化策略參數
        strategy = PairsTradingStrategy(
            lookback_period=40,  # 增加回顧期
            enter_long_zscore_threshold=1.5,  # 調整入場門檻
            enter_short_zscore_threshold=1.5,
            exit_zscore_threshold=0.5,
            min_samples=60,
            coint_pvalue=0.05,
            min_correlation=0.8
        )
        
        # 執行優化後的策略
        print("Applying enhanced stock filtering...")
        valid_stocks = enhanced_filter_stocks(all_stocks_daily)
        print(f"Found {len(valid_stocks)} valid stocks after filtering")
        
        print("\nSelecting optimal pairs...")
        best_pairs = enhanced_pair_selection(all_stocks_daily, valid_stocks)
        print(f"Found {len(best_pairs)} potential pairs")
        
        # 執行回測
        results = []
        for pair_info in best_pairs:
            stock1, stock2, corr, p_value, score = pair_info
            print(f"\nTesting pair {stock1}-{stock2}:")
            print(f"Correlation: {corr:.3f}")
            print(f"Cointegration p-value: {p_value:.3f}")
            print(f"Pair score: {score:.3f}")
            
            try:
                result = strategy.execute_pair_trade(
                    all_stocks_daily[stock1],
                    all_stocks_daily[stock2],
                    (stock1, stock2)
                )
                
                if result is not None and result.metrics['win_rate'] > 0.4:  # 提高勝率要求
                    results.append(result)
                    print("Trading metrics:")
                    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
                    print(f"Total Return: {result.metrics['total_return']:.2%}")
                    print(f"Win Rate: {result.metrics['win_rate']:.2%}")
                    
            except Exception as e:
                print(f"Error processing pair: {str(e)}")
                continue
        
        # 結果分析
        if results:
            results_df = pd.DataFrame([{
                'pair': f"{r.pair[0]}-{r.pair[1]}",
                'start_date': r.start_date,
                'end_date': r.end_date,
                **r.metrics
            } for r in results])
            
            # 篩選表現好的配對
            results_df = results_df[
                (results_df['sharpe_ratio'] > 0.5) &  # 提高夏普比率要求
                (results_df['win_rate'] > 0.4) &      # 提高勝率要求
                (results_df['total_return'] > 0)       # 要求正收益
            ]
            
            if not results_df.empty:
                print("\nFiltered Results Summary:")
                print("\nTop pairs by performance:")
                print(results_df[['pair', 'sharpe_ratio', 'total_return', 'win_rate']])
            else:
                print("\nNo pairs met the enhanced performance criteria")
                
    except Exception as e:
        print(f"Error in test execution: {str(e)}")
        raise

if __name__ == "__main__":
    test_with_sample_data()