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
import json
import logging

# Set up logging
logging.basicConfig(
    filename='pairs_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class PairTradingResult:
    pair: Tuple[str, str]
    start_date: str
    end_date: str
    positions: pd.DataFrame
    returns: pd.Series
    metrics: Dict
    exposures: pd.DataFrame

class ConfigLoader:
    @staticmethod
    def load_config(file_path: str) -> dict:
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config

#################################################################################################

class PairsDataProcessor:
    def __init__(self, data_folder: Path):
        self.data_folder = data_folder
        
    def load_stock_data(self, stock_code: str, file_path: Path) -> Optional[pd.DataFrame]:
        try:
            stock_df = pd.read_csv(
                file_path,
                parse_dates=['ts'],
                usecols=['ts', 'Close'],
                dtype={'Close': 'float32'}
            )
            
            if stock_df.empty:
                return None
                
            stock_df.set_index('ts', inplace=True)
            return stock_df
            
        except Exception as e:
            logging.error(f"Error loading {stock_code}: {str(e)}")
            return None
    
    def resample_to_daily(self, 
                         stock_df: pd.DataFrame,
                         trading_days: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        try:
            if stock_df is None or stock_df.empty:
                return None
            
            daily_df = stock_df.resample('D').last()
            daily_df = daily_df.reindex(trading_days)
            return daily_df
            
        except Exception as e:
            logging.error(f"Error resampling data to daily: {str(e)}")
            return None
    
    def combine_stock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        all_stocks_daily = {}
        
        trading_days = mcal.get_calendar('XTAI').schedule(
            start_date=start_date, 
            end_date=end_date
        ).index
        
        csv_files = list(self.data_folder.glob('*.csv'))
        total_files = len(csv_files)
        logging.info(f"Found {total_files} CSV files")
        
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
                        
                    if completed % 10 == 0:
                        progress = (completed / total_files) * 100
                        logging.info(f"Loading progress: {progress:.1f}% ({completed}/{total_files} files)")
                        
                except Exception as e:
                    logging.error(f"Error processing {stock_code}: {str(e)}")
                    continue
        
        logging.info(f"Successfully loaded {len(all_stocks_daily)} stocks")
        
        df = pd.DataFrame(all_stocks_daily)
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def load_and_process_stock(self, csv_file: Path, trading_days: pd.DatetimeIndex) -> Optional[pd.Series]:
        stock_df = self.load_stock_data(csv_file.stem, csv_file)
        if stock_df is None:
            return None
            
        daily_df = self.resample_to_daily(stock_df, trading_days)
        if daily_df is None:
            return None
            
        return daily_df['Close']

#########################################################################

class PairsTradingStrategy:
    def __init__(self, config_path: str):
        config = ConfigLoader.load_config(config_path)
        self.lookback_period = config['lookback_period']
        self.enter_long_zscore_threshold = config['enter_long_zscore_threshold']
        self.enter_short_zscore_threshold = config['enter_short_zscore_threshold']
        self.exit_long_zscore_threshold = config['exit_long_zscore_threshold']
        self.exit_short_zscore_threshold = config['exit_short_zscore_threshold']
        self.min_samples = config['min_samples']
        self.coint_pvalue = config['coint_pvalue']
        self.min_correlation = config['min_correlation']
        self.transaction_cost = config['transaction_cost']
    
    def prepare_pair_data(self, y: pd.Series, x: pd.Series) -> Tuple[pd.Series, pd.Series]:
        mask = ~np.isnan(y) & ~np.isnan(x)
        return y[mask], x[mask]
    
    def calculate_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        return np.cov(y, x)[0, 1] / np.var(x)
    
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        mean = spread.rolling(window=self.lookback_period).mean()
        std = spread.rolling(window=self.lookback_period).std()
        return (spread - mean) / std
    
    def generate_signals(self, zscore: pd.Series) -> pd.Series:
        signals = pd.Series(0, index=zscore.index)
        position = 0
        
        for i in range(len(zscore)):
            if position == 0:
                if zscore.iloc[i] < -self.enter_long_zscore_threshold:
                    position = 1
                    signals.iloc[i] = 1
                elif zscore.iloc[i] > self.enter_short_zscore_threshold:
                    position = -1
                    signals.iloc[i] = -1
            elif position == 1:
                if zscore.iloc[i] >= self.exit_long_zscore_threshold:
                    position = 0
                    signals.iloc[i] = 0
            elif position == -1:
                if zscore.iloc[i] <= -self.exit_short_zscore_threshold:
                    position = 0
                    signals.iloc[i] = 0
                    
        return signals
    
    def calculate_returns(self, 
                         pair_data: pd.DataFrame, 
                         signals: pd.Series, 
                         hedge_ratio: float) -> pd.Series:
        stock1_rets = pair_data.iloc[:, 0].pct_change()
        stock2_rets = pair_data.iloc[:, 1].pct_change()
        pos_changes = signals.diff().fillna(0)
        
        stock1_notional = 1.0
        stock2_notional = hedge_ratio
        total_notional = abs(stock1_notional) + abs(stock2_notional)
        
        stock1_weight = stock1_notional / total_notional
        stock2_weight = stock2_notional / total_notional
        
        strategy_rets = signals.shift(1) * (
            stock1_weight * stock1_rets - 
            stock2_weight * stock2_rets
        )
        
        transaction_costs = abs(pos_changes) * self.transaction_cost * (
            abs(stock1_weight) + abs(stock2_weight)
        )
        
        strategy_rets = strategy_rets - transaction_costs
        
        return strategy_rets
    
    def calculate_position_exposures(self,
                                   pair_data: pd.DataFrame,
                                   signals: pd.Series,
                                   hedge_ratio: float) -> pd.DataFrame:
        stock1_notional = 1.0
        stock2_notional = hedge_ratio
        total_notional = abs(stock1_notional) + abs(stock2_notional)
        
        stock1_weight = stock1_notional / total_notional
        stock2_weight = stock2_notional / total_notional
        
        stock1_position = signals * stock1_weight
        stock2_position = -signals * stock2_weight
        
        stock1_exposure = stock1_position * pair_data.iloc[:, 0]
        stock2_exposure = stock2_position * pair_data.iloc[:, 1]
        
        return pd.DataFrame({
            'stock1_position': stock1_position,
            'stock2_position': stock2_position,
            'stock1_exposure': stock1_exposure,
            'stock2_exposure': stock2_exposure,
            'net_exposure': stock1_exposure + stock2_exposure,  
            'gross_exposure': abs(stock1_exposure) + abs(stock2_exposure)
        })

    def calculate_metrics(self, returns: pd.Series) -> Dict:
        annual_factor = 252
        
        total_return = np.expm1(np.sum(np.log1p(returns)))
        annual_return = np.expm1(np.sum(np.log1p(returns)) * annual_factor / len(returns))
        annual_volatility = returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = np.sqrt(annual_factor) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
        winning_trades = (returns > 0).sum()
        total_trades = (~returns.isna()).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        cumulative_returns = (1 + returns).cumprod() - 1
        drawdowns = cumulative_returns - cumulative_returns.cummax()
        max_drawdown = drawdowns.min()
        
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
        
        daily_pnl = returns.sum()
        avg_daily_pnl = returns.mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'total_pnl': daily_pnl,
            'average_daily_pnl': avg_daily_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }
    
    def check_pair_validity(self, y: pd.Series, x: pd.Series) -> bool:
        if len(y) < self.min_samples:
            return False
            
        correlation = np.corrcoef(y, x)[0, 1]
        if abs(correlation) < self.min_correlation:
            return False
            
        try:
            _, p_value, _ = coint(y, x)
            return p_value <= self.coint_pvalue
        except Exception as e:
            logging.error(f"Error in cointegration test: {str(e)}")
            return False

    def execute_pair_trade(self, 
                      stock1_data: pd.Series, 
                      stock2_data: pd.Series,
                      pair: Tuple[str, str]) -> Optional[PairTradingResult]:
        try:
            stock1_clean, stock2_clean = self.prepare_pair_data(stock1_data, stock2_data)
            
            if stock1_clean.empty or stock2_clean.empty:
                logging.warning(f"Empty data for pair {pair}")
                return None
            
            if len(stock1_clean) != len(stock2_clean):
                logging.warning(f"Mismatched data lengths for pair {pair}")
                return None
            
            if not self.check_pair_validity(stock1_clean, stock2_clean):
                return None
            
            hedge_ratio = self.calculate_hedge_ratio(stock1_clean, stock2_clean)
            spread = stock1_clean - hedge_ratio * stock2_clean
            
            zscore = self.calculate_zscore(spread)
            signals = self.generate_signals(zscore)
            
            pair_data = pd.concat([stock1_clean, stock2_clean], axis=1)
            pair_data.columns = [f"{pair[0]}_price", f"{pair[1]}_price"]
            
            returns = self.calculate_returns(pair_data, signals, hedge_ratio)
            exposures = self.calculate_position_exposures(pair_data, signals, hedge_ratio)
            
            trade_changes = signals.diff().fillna(0)
            trade_entries = trade_changes != 0
            trade_count = trade_entries.sum()
            
            if trade_count == 0:
                logging.info(f"No trades generated for pair {pair}")
                return None
            
            metrics = self.calculate_metrics(returns.dropna())
            
            metrics.update({
                'number_of_trades': trade_count,
                'avg_trade_duration': len(signals) / trade_count if trade_count > 0 else 0,
                'hedge_ratio': hedge_ratio,
                'spread_stdev': spread.std(),
                'correlation': np.corrcoef(stock1_clean, stock2_clean)[0, 1]
            })
            
            positions = pd.DataFrame({
                'signals': signals,
                'zscore': zscore,
                'spread': spread,
                'stock1_price': pair_data.iloc[:, 0],
                'stock2_price': pair_data.iloc[:, 1],
                'stock1_position': exposures['stock1_position'],
                'stock2_position': exposures['stock2_position'],
                'net_exposure': exposures['net_exposure']
            })
            
            positions['trade_entry'] = trade_entries
            positions['trade_exit'] = trade_changes != 0
            
            cumulative_returns = (1 + returns).cumprod()
            drawdown_series = cumulative_returns / cumulative_returns.cummax() - 1
            positions['drawdown'] = drawdown_series
            
            result = PairTradingResult(
                pair=pair,
                start_date=stock1_clean.index[0].strftime('%Y-%m-%d'),
                end_date=stock1_clean.index[-1].strftime('%Y-%m-%d'),
                positions=positions,
                returns=returns,
                metrics=metrics,
                exposures=exposures
            )
            
            result.hedge_ratio = hedge_ratio
            result.spread_mean = spread.mean()
            result.spread_std = spread.std()
            result.trade_count = trade_count
            result.drawdown_series = drawdown_series
            
            logging.info(f"Successfully processed pair {pair} with {trade_count} trades")
            return result
            
        except Exception as e:
            logging.error(f"Error processing pair {pair}: {str(e)}", exc_info=True)
            return None
