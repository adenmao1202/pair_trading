# main_pairs_trading.py

import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas_market_calendars as mcal
from basic_coint import PairsDataProcessor, PairsTradingStrategy, PairTradingResult
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_trading_frequency(results_df: pd.DataFrame) -> pd.DataFrame:
    """分析交易頻率模式"""
    frequency_analysis = pd.DataFrame()
    
    # 計算平均交易間隔
    frequency_analysis['avg_days_between_trades'] = results_df.apply(
        lambda x: (pd.to_datetime(x['end_date']) - pd.to_datetime(x['start_date'])).days / 
        (x['winning_trades'] + x['losing_trades']), axis=1
    )
    
    # 計算每月交易次數
    frequency_analysis['trades_per_month'] = results_df.apply(
        lambda x: (x['winning_trades'] + x['losing_trades']) * 30 / 
        (pd.to_datetime(x['end_date']) - pd.to_datetime(x['start_date'])).days, axis=1
    )
    
    return frequency_analysis

def analyze_spreads(results: List[PairTradingResult]) -> pd.DataFrame:
    """分析價差特徵"""
    spread_analysis = pd.DataFrame([{
        'pair': f"{r.pair[0]}-{r.pair[1]}",
        'spread_mean': r.spread_mean,
        'spread_std': r.spread_std,
        'zscore_mean': r.zscore_mean,
        'zscore_std': r.zscore_std
    } for r in results])
    
    return spread_analysis

def analyze_trade_characteristics(results_df: pd.DataFrame) -> pd.DataFrame:
    """分析交易特徵"""
    trade_analysis = pd.DataFrame()
    
    # 勝率與獲利比率的關係
    trade_analysis['win_profit_ratio'] = results_df['win_rate'] * results_df['average_win'].abs()
    trade_analysis['loss_risk_ratio'] = (1 - results_df['win_rate']) * results_df['average_loss'].abs()
    trade_analysis['risk_reward_ratio'] = trade_analysis['win_profit_ratio'] / trade_analysis['loss_risk_ratio']
    
    return trade_analysis

def calculate_pair_correlations(price_data: pd.DataFrame) -> pd.DataFrame:
    """計算配對相關性矩陣"""
    return price_data.corr()

def save_detailed_results(results: List[PairTradingResult], 
                         results_df: pd.DataFrame, 
                         output_folder: Path):
    """保存詳細的交易結果"""
    # 創建輸出目錄
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_folder = output_folder / f'results_{timestamp}'
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # 保存主要結果
    results_df.to_csv(results_folder / 'all_pairs_results.csv', index=False)
    
    # 創建詳細分析目錄
    detailed_folder = results_folder / 'detailed_analysis'
    detailed_folder.mkdir(exist_ok=True)
    
    # 保存每個配對的詳細資訊
    for result in results:
        pair_name = f"{result.pair[0]}-{result.pair[1]}"
        pair_folder = detailed_folder / pair_name
        pair_folder.mkdir(exist_ok=True)
        
        # 保存交易位置
        result.positions.to_csv(pair_folder / 'positions.csv')
        
        # 保存收益率
        pd.Series(result.returns).to_csv(pair_folder / 'returns.csv')
        
        # 保存部位曝險
        result.exposures.to_csv(pair_folder / 'exposures.csv')
        
        # 生成圖表
        generate_pair_analysis_plots(result, pair_folder)

def generate_pair_analysis_plots(result: PairTradingResult, output_folder: Path):
    """為每個配對生成分析圖表"""
    # 價格和訊號圖
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(result.positions.index, result.positions['stock1_price'], label='Stock 1')
    plt.plot(result.positions.index, result.positions['stock2_price'], label='Stock 2')
    plt.title(f"Price Movement for {result.pair[0]}-{result.pair[1]}")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(result.positions.index, result.positions['signals'], label='Trading Signals')
    plt.title('Trading Signals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / 'price_signals.png')
    plt.close()
    
    # 收益率和回撤圖
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(result.positions.index, result.positions['drawdown'], label='Drawdown')
    plt.title('Drawdown Analysis')
    plt.legend()
    
    cumulative_returns = (1 + pd.Series(result.returns)).cumprod() - 1
    plt.subplot(2, 1, 2)
    plt.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / 'returns_analysis.png')
    plt.close()

def format_top_pairs_output(top_pairs: pd.DataFrame) -> pd.DataFrame:
    """格式化頂級配對的輸出"""
    formatted_df = top_pairs.copy()
    
    # 設定顯示的列
    display_columns = [
        'pair',
        'calmar_ratio',
        'annual_return',
        'max_drawdown',
        'winning_trades',
        'losing_trades',
        'average_win',
        'average_loss',
        'win_rate',
        'profit_factor'
    ]
    
    # 重命名列
    column_names = {
        'pair': 'Pair',
        'calmar_ratio': 'Calmar Ratio',
        'annual_return': 'Annual Return',
        'max_drawdown': 'Max Drawdown',
        'winning_trades': 'Winning Trades',
        'losing_trades': 'Losing Trades',
        'average_win': 'Avg Win',
        'average_loss': 'Avg Loss',
        'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor'
    }
    
    formatted_df = formatted_df[display_columns].rename(columns=column_names)
    return formatted_df

def main():
    """主要執行函數"""
    # 設定
    DATA_FOLDER = Path('/Users/mouyasushi/k_data/永豐')
    OUTPUT_FOLDER = Path('/Users/mouyasushi/Desktop/pair_trading/output')
    START_DATE = '2022-10-14'
    END_DATE = '2024-10-14'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUTPUT_FOLDER = OUTPUT_FOLDER / f'analysis_{timestamp}'
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    try:
        # 初始化處理器和策略
        processor = PairsDataProcessor(DATA_FOLDER)
        strategy = PairsTradingStrategy()
        
        # 載入和處理數據
        print("Loading and processing data...")
        all_stocks_daily = processor.combine_stock_data(START_DATE, END_DATE)
        
        # 執行策略
        print("Executing pairs trading strategy...")
        results = []
        stock_codes = all_stocks_daily.columns
        pairs = [(stock1, stock2) 
                for i, stock1 in enumerate(stock_codes) 
                for stock2 in stock_codes[i+1:]]
        
        # 使用進度條處理配對
        with tqdm(total=len(pairs), desc="Processing pairs") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                future_to_pair = {
                    executor.submit(
                        strategy.execute_pair_trade,
                        all_stocks_daily[pair[0]],
                        all_stocks_daily[pair[1]],
                        pair
                    ): pair for pair in pairs
                }
                
                for future in concurrent.futures.as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error with pair {pair}: {str(e)}")
                    pbar.update(1)
        
        if not results:
            print("No valid pairs found.")
            return
            
        # 建立結果DataFrame
        results_df = pd.DataFrame([{
            'pair': f"{r.pair[0]}-{r.pair[1]}",
            'start_date': r.start_date,
            'end_date': r.end_date,
            **r.metrics
        } for r in results])
        
        # 計算額外指標
        results_df['calmar_ratio'] = results_df.apply(lambda row: 
            abs(row['annual_return']) / abs(row['max_drawdown']) 
            if row['max_drawdown'] != 0 else float('inf'), axis=1)
        
        # 進行額外分析
        trading_frequency = analyze_trading_frequency(results_df)
        spread_analysis = analyze_spreads(results)
        trade_characteristics = analyze_trade_characteristics(results_df)
        
        # 保存分析結果
        analysis_folder = OUTPUT_FOLDER / 'analysis'
        analysis_folder.mkdir(exist_ok=True)
        
        trading_frequency.to_csv(analysis_folder / 'trading_frequency.csv')
        spread_analysis.to_csv(analysis_folder / 'spread_analysis.csv')
        trade_characteristics.to_csv(analysis_folder / 'trade_characteristics.csv')
        
        # 獲取並顯示前20名配對
        print("\nTop 20 Pairs by Calmar Ratio:")
        print("=" * 120)
        top_20_pairs = results_df.nlargest(20, 'calmar_ratio')
        formatted_top_20 = format_top_pairs_output(top_20_pairs)
        print(formatted_top_20.to_string(
            formatters={
                'Calmar Ratio': '{:>8.2f}'.format,
                'Annual Return': '{:>8.2%}'.format,
                'Max Drawdown': '{:>8.2%}'.format,
                'Avg Win': '{:>8.2%}'.format,
                'Avg Loss': '{:>8.2%}'.format,
                'Win Rate': '{:>8.2%}'.format,
                'Profit Factor': '{:>6.2f}'.format
            }
        ))
        print("=" * 120)
        
        # 保存詳細結果
        save_detailed_results(results, results_df, OUTPUT_FOLDER)
        
        print(f"\nAnalysis completed. Results saved to {OUTPUT_FOLDER}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()