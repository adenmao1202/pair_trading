import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas_market_calendars as mcal
from datetime import datetime
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
import concurrent.futures
from dataclasses import dataclass

@dataclass
class PairTradingResult:
    pair: Tuple[str, str]
    start_date: str
    end_date: str
    positions: pd.DataFrame
    returns: pd.Series
    metrics: Dict

class PairsDataProcessor:
    def __init__(self, data_folder: Path):
        self.data_folder = data_folder
        
    def load_stock_data(self, stock_code: str, file_path: Path) -> Optional[pd.DataFrame]:
        """Read and process individual stock data with error handling."""
        try:
            # 只讀取必要的欄位來減少記憶體使用
            stock_df = pd.read_csv(
                file_path,
                parse_dates=['ts'],
                usecols=['ts', 'Close'],  # 只讀取需要的欄位
                dtype={'Close': 'float32'}  # 使用較小的數據類型
            )
            
            if stock_df.empty:
                return None
                
            stock_df.set_index('ts', inplace=True)
            return stock_df
            
        except Exception as e:
            print(f"Error loading {stock_code}: {str(e)}")
            return None
    
    def resample_to_daily(self, 
                         stock_df: pd.DataFrame,
                         trading_days: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        """Optimized daily resampling."""
        try:
            if stock_df is None or stock_df.empty:
                return None
            
            # 直接使用last而不是複雜的聚合
            daily_df = stock_df.resample('D').last()
            daily_df = daily_df.reindex(trading_days)
            return daily_df
            
        except Exception:
            return None
    
    def combine_stock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """優化的股票數據處理"""
        all_stocks_daily = {}
        
        # 預先獲取交易日曆
        trading_days = mcal.get_calendar('XTAI').schedule(
            start_date=start_date, 
            end_date=end_date
        ).index
        
        # 獲取所有CSV文件
        csv_files = list(self.data_folder.glob('*.csv'))
        total_files = len(csv_files)
        print(f"Found {total_files} CSV files")
        
        # 使用較少的線程數進行並行處理
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_stock = {
                executor.submit(self.load_and_process_stock, csv_file, trading_days): csv_file.stem
                for csv_file in csv_files
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result is not None:
                        all_stocks_daily[stock_code] = result
                        
                    # 顯示進度
                    if completed % 10 == 0:
                        progress = (completed / total_files) * 100
                        print(f"Loading progress: {progress:.1f}% ({completed}/{total_files} files)")
                        
                except Exception as e:
                    print(f"Error processing {stock_code}: {str(e)}")
                    continue
        
        print(f"\nSuccessfully loaded {len(all_stocks_daily)} stocks")
        
        # 轉換為DataFrame並確保數據完整性
        df = pd.DataFrame(all_stocks_daily)
        
        # 移除過多缺失值的列
        threshold = len(df.columns) * 0.5  # 如果超過50%的數據缺失，則移除該行
        df = df.dropna(thresh=threshold)
        
        # 用前向填充方法處理剩餘的缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def load_and_process_stock(self, csv_file: Path, trading_days: pd.DatetimeIndex) -> Optional[pd.Series]:
        """Helper function for parallel processing."""
        stock_df = self.load_stock_data(csv_file.stem, csv_file)
        if stock_df is None:
            return None
            
        daily_df = self.resample_to_daily(stock_df, trading_days)
        if daily_df is None:
            return None
            
        return daily_df['Close']

class PairsTradingStrategy:
    def __init__(self, 
                 lookback_period: int = 60,
                 zscore_threshold: float = 2.0,
                 min_samples: int = 252,
                 coint_pvalue: float = 0.10,
                 min_correlation: float = 0.7):
        self.lookback_period = lookback_period
        self.zscore_threshold = zscore_threshold
        self.min_samples = min_samples
        self.coint_pvalue = coint_pvalue
        self.min_correlation = min_correlation
    
    def prepare_pair_data(self, y: pd.Series, x: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """準備並清理配對數據"""
        # 使用向量化操作而不是循環
        mask = ~np.isnan(y) & ~np.isnan(x)
        return y[mask], x[mask]
    
    def calculate_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        """使用NumPy進行快速線性回歸"""
        return np.cov(y, x)[0, 1] / np.var(x)
    
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """優化的z分數計算"""
        # 使用向量化操作
        mean = pd.Series(index=spread.index, dtype='float32')
        std = pd.Series(index=spread.index, dtype='float32')
        
        for i in range(self.lookback_period, len(spread)):
            window = spread.iloc[i-self.lookback_period:i]
            mean.iloc[i] = window.mean()
            std.iloc[i] = window.std()
            
        return (spread - mean) / std
    
    def generate_signals(self, zscore: pd.Series) -> pd.Series:
        """優化的信號生成"""
        signals = pd.Series(0, index=zscore.index)
        position = 0
        
        # 使用NumPy的向量化操作
        long_entries = (zscore < -self.zscore_threshold) & (position == 0)
        short_entries = (zscore > self.zscore_threshold) & (position == 0)
        long_exits = (zscore >= 0) & (position == 1)
        short_exits = (zscore <= 0) & (position == -1)
        
        signals[long_entries] = 1
        signals[short_entries] = -1
        signals[long_exits | short_exits] = 0
        
        return signals
    
    def calculate_returns(self, 
                         pair_data: pd.DataFrame, 
                         signals: pd.Series, 
                         hedge_ratio: float,
                         transaction_cost: float = 0.001) -> pd.Series:
        """優化的收益計算"""
        # 使用向量化操作
        stock1_rets = pair_data.iloc[:, 0].pct_change()
        stock2_rets = pair_data.iloc[:, 1].pct_change()
        pos_changes = signals.diff().fillna(0)
        
        # 一次性計算所有收益
        strategy_rets = signals.shift(1) * (stock1_rets - hedge_ratio * stock2_rets) - \
                       abs(pos_changes) * transaction_cost
        
        return strategy_rets
    
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """優化的績效指標計算"""
        annual_factor = 252
        
        # 使用NumPy的向量化操作
        total_return = np.expm1(np.sum(np.log1p(returns)))
        annual_return = np.expm1(np.sum(np.log1p(returns)) * annual_factor / len(returns))
        annual_volatility = returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = np.sqrt(annual_factor) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def check_pair_validity(self, y: pd.Series, x: pd.Series) -> bool:
        """簡化的配對有效性檢查"""
        if len(y) < self.min_samples:
            return False
            
        correlation = np.corrcoef(y, x)[0, 1]
        if abs(correlation) < self.min_correlation:
            return False
            
        try:
            _, p_value, _ = coint(y, x)
            return p_value <= self.coint_pvalue
        except:
            return False

    def execute_pair_trade(self, 
                          stock1_data: pd.Series, 
                          stock2_data: pd.Series,
                          pair: Tuple[str, str]) -> Optional[PairTradingResult]:
        """優化的配對交易執行"""
        try:
            # 清理並準備數據
            stock1_clean, stock2_clean = self.prepare_pair_data(stock1_data, stock2_data)
            
            # 快速檢查有效性
            if not self.check_pair_validity(stock1_clean, stock2_clean):
                return None
                
            # 計算對沖比率和價差
            hedge_ratio = self.calculate_hedge_ratio(stock1_clean, stock2_clean)
            spread = stock1_clean - hedge_ratio * stock2_clean
            
            # 生成信號和計算收益
            zscore = self.calculate_zscore(spread)
            signals = self.generate_signals(zscore)
            pair_data = pd.concat([stock1_clean, stock2_clean], axis=1)
            returns = self.calculate_returns(pair_data, signals, hedge_ratio)
            
            # 計算績效指標
            metrics = self.calculate_metrics(returns.dropna())
            
            return PairTradingResult(
                pair=pair,
                start_date=stock1_clean.index[0].strftime('%Y-%m-%d'),
                end_date=stock1_clean.index[-1].strftime('%Y-%m-%d'),
                positions=pd.DataFrame({'signals': signals, 'zscore': zscore}),
                returns=returns,
                metrics=metrics
            )
            
        except Exception as e:
            print(f"Error processing pair {pair}: {str(e)}")
            return None

def main():
    # 配置
    DATA_FOLDER = Path('/Users/mouyasushi/k_data/永豐')
    OUTPUT_FOLDER = Path('/Users/mouyasushi/Desktop/pair_trading/output')
    START_DATE = '2022-10-14'
    END_DATE = '2024-10-14'
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # 初始化處理器和策略
    processor = PairsDataProcessor(DATA_FOLDER)
    strategy = PairsTradingStrategy()
    
    try:
        # 載入和處理數據
        print("Loading and processing data...")
        all_stocks_daily = processor.combine_stock_data(START_DATE, END_DATE)
        
        # 執行策略
        print("Executing pairs trading strategy...")
        results = []
        stock_codes = all_stocks_daily.columns
        total_pairs = len(stock_codes) * (len(stock_codes) - 1) // 2
        processed_pairs = 0
        
        # 批次處理配對
        batch_size = 100  # 每批處理的配對數量
        pairs = []
        
        # 生成所有配對組合
        for i, stock1 in enumerate(stock_codes):
            for stock2 in stock_codes[i+1:]:
                pairs.append((stock1, stock2))
        
        # 按批次處理
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # 使用較少的worker數量
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                future_to_pair = {
                    executor.submit(
                        strategy.execute_pair_trade,
                        all_stocks_daily[pair[0]],
                        all_stocks_daily[pair[1]],
                        pair
                    ): pair for pair in batch_pairs
                }
                
                for future in concurrent.futures.as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    processed_pairs += 1
                    
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            
                        # 顯示進度
                        if processed_pairs % 10 == 0:
                            progress = (processed_pairs / total_pairs) * 100
                            print(f"Progress: {progress:.1f}% ({processed_pairs}/{total_pairs} pairs)")
                            
                    except Exception as e:
                        print(f"Error with pair {pair}: {str(e)}")
        
        print(f"\nProcessing complete. Found {len(results)} valid pairs out of {total_pairs} total pairs.")
        
        if not results:
            print("No valid pairs found.")
            return
        
        # 儲存結果
        results_df = pd.DataFrame([{
            'pair': f"{r.pair[0]}-{r.pair[1]}",
            'start_date': r.start_date,
            'end_date': r.end_date,
            **r.metrics
        } for r in results])
        
        # 儲存中間結果
        results_df.to_csv(OUTPUT_FOLDER / 'pairs_trading_results.csv', index=False)
        
        # 顯示最佳配對
        best_pairs = results_df.nlargest(5, 'sharpe_ratio')
        print("\nTop 5 Pairs by Sharpe Ratio:")
        print(best_pairs)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()