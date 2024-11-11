# main_pairs_trading.py

import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas_market_calendars as mcal
from basicoint_config import PairsDataProcessor, PairsTradingStrategy, PairTradingResult, ConfigLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def analyze_trading_frequency(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze trading frequency patterns"""
    frequency_analysis = pd.DataFrame()
    
    # Calculate average interval between trades
    frequency_analysis['avg_days_between_trades'] = results_df.apply(
        lambda x: (pd.to_datetime(x['end_date']) - pd.to_datetime(x['start_date'])).days / 
        x['metrics']['total_trades'] if x['metrics']['total_trades'] > 0 else np.nan, axis=1
    )
    
    # Calculate trades per month
    frequency_analysis['trades_per_month'] = results_df.apply(
        lambda x: x['metrics']['total_trades'] * 30 / 
        (pd.to_datetime(x['end_date']) - pd.to_datetime(x['start_date'])).days, axis=1
    )
    
    return frequency_analysis

def analyze_spreads(results: List[PairTradingResult]) -> pd.DataFrame:
    """Analyze spread characteristics"""
    spread_analysis = pd.DataFrame([{
        'pair': f"{r.pair[0]}-{r.pair[1]}",
        'hedge_ratio': r.hedge_ratio,
        'spread_mean': r.spread_mean,
        'spread_std': r.spread_std,
        'trade_count': r.trade_count
    } for r in results])
    
    return spread_analysis

def analyze_trade_characteristics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze trading characteristics"""
    trade_analysis = pd.DataFrame()
    
    # Calculate win/loss ratios and risk metrics
    trade_analysis['win_profit_ratio'] = results_df.apply(
        lambda x: x['metrics']['win_rate'] * abs(x['metrics']['average_win']), axis=1
    )
    trade_analysis['loss_risk_ratio'] = results_df.apply(
        lambda x: (1 - x['metrics']['win_rate']) * abs(x['metrics']['average_loss']), axis=1
    )
    trade_analysis['risk_reward_ratio'] = trade_analysis['win_profit_ratio'] / trade_analysis['loss_risk_ratio'].replace(0, np.inf)
    
    return trade_analysis

def generate_pair_analysis_plots(result: PairTradingResult, output_folder: Path):
    """Generate analysis plots for each pair"""
    # Price and signals plot
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
    
    # Returns and drawdown plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(result.positions.index, result.positions['drawdown'], label='Drawdown')
    plt.title('Drawdown Analysis')
    plt.legend()
    
    cumulative_returns = (1 + result.returns).cumprod() - 1
    plt.subplot(2, 1, 2)
    plt.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / 'returns_analysis.png')
    plt.close()


def analyze_best_pairs(results: List[PairTradingResult], top_n: int = 20) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Analyze and return detailed results for the best performing pairs based on multiple metrics
    
    Args:
        results: List of PairTradingResult objects
        top_n: Number of top pairs to analyze
    
    Returns:
        summary_df: Summary DataFrame of all pairs
        detailed_results: Dictionary containing detailed analysis DataFrames
    """
    # Create comprehensive summary DataFrame
    summary_data = []
    for r in results:
        pair_name = f"{r.pair[0]}-{r.pair[1]}"
        metrics = r.metrics
        
        summary_data.append({
            'pair': pair_name,
            'total_trades': metrics['total_trades'],
            'winning_trades': metrics['winning_trades'],
            'losing_trades': metrics['losing_trades'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'total_return': metrics['total_return'],
            'annual_return': metrics['annual_return'],
            'annual_volatility': metrics['annual_volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'calmar_ratio': abs(metrics['annual_return']) / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else float('inf'),
            'max_drawdown': metrics['max_drawdown'],
            'avg_win': metrics['average_win'],
            'avg_loss': metrics['average_loss'],
            'total_pnl': metrics['total_pnl'],
            'avg_trade_duration': metrics['avg_trade_duration'] if 'avg_trade_duration' in metrics else None,
            'start_date': r.start_date,
            'end_date': r.end_date
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Get top pairs by both Sharpe and Calmar ratios
    top_sharpe = set(summary_df.nlargest(top_n, 'sharpe_ratio')['pair'])
    top_calmar = set(summary_df.nlargest(top_n, 'calmar_ratio')['pair'])
    top_pairs = list(top_sharpe.union(top_calmar))
    
    # Create detailed analysis for top pairs
    detailed_results = {}
    for pair in top_pairs:
        result = next(r for r in results if f"{r.pair[0]}-{r.pair[1]}" == pair)
        
        # Trading statistics
        trade_stats = pd.DataFrame({
            'entry_date': result.positions[result.positions['trade_entry']].index,
            'exit_date': result.positions[result.positions['trade_exit']].index,
            'position': result.positions['signals'][result.positions['trade_entry']],
            'entry_spread': result.positions['spread'][result.positions['trade_entry']],
            'exit_spread': result.positions['spread'][result.positions['trade_exit']],
            'trade_pnl': result.returns[result.positions['trade_exit']],
            'trade_duration': (result.positions[result.positions['trade_exit']].index - 
                             result.positions[result.positions['trade_entry']].index).days
        }).reset_index(drop=True)
        
        # Performance metrics over time
        performance_metrics = pd.DataFrame({
            'cumulative_returns': (1 + result.returns).cumprod() - 1,
            'drawdown': result.positions['drawdown'],
            'rolling_sharpe': result.returns.rolling(252).mean() / result.returns.rolling(252).std() * np.sqrt(252),
            'rolling_volatility': result.returns.rolling(252).std() * np.sqrt(252)
        })
        
        # Position and exposure analysis
        position_analysis = pd.DataFrame({
            'signals': result.positions['signals'],
            'spread': result.positions['spread'],
            'zscore': result.positions['zscore'],
            'stock1_exposure': result.exposures['stock1_exposure'],
            'stock2_exposure': result.exposures['stock2_exposure'],
            'net_exposure': result.exposures['net_exposure'],
            'gross_exposure': result.exposures['gross_exposure']
        })
        
        detailed_results[pair] = {
            'trade_stats': trade_stats,
            'performance_metrics': performance_metrics,
            'position_analysis': position_analysis
        }
    
    return summary_df, detailed_results

def generate_performance_report(summary_df: pd.DataFrame, detailed_results: Dict[str, pd.DataFrame], output_folder: Path):
    """Generate comprehensive performance report for top pairs"""
    # Create performance report directory
    report_folder = output_folder / 'performance_report'
    report_folder.mkdir(parents=True, exist_ok=True)
    
    # Summary of top pairs by different metrics
    print("\nTop Pairs by Sharpe Ratio:")
    print("=" * 120)
    print(summary_df.nlargest(20, 'sharpe_ratio')[
        ['pair', 'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 
         'profit_factor', 'total_trades', 'total_pnl']
    ].to_string(
        formatters={
            'sharpe_ratio': '{:,.2f}'.format,
            'total_return': '{:,.2%}'.format,
            'max_drawdown': '{:,.2%}'.format,
            'win_rate': '{:,.2%}'.format,
            'profit_factor': '{:,.2f}'.format,
            'total_pnl': '{:,.2f}'.format
        }
    ))
    
    print("\nTop Pairs by Calmar Ratio:")
    print("=" * 120)
    print(summary_df.nlargest(20, 'calmar_ratio')[
        ['pair', 'calmar_ratio', 'annual_return', 'max_drawdown', 'win_rate', 
         'profit_factor', 'total_trades', 'total_pnl']
    ].to_string(
        formatters={
            'calmar_ratio': '{:,.2f}'.format,
            'annual_return': '{:,.2%}'.format,
            'max_drawdown': '{:,.2%}'.format,
            'win_rate': '{:,.2%}'.format,
            'profit_factor': '{:,.2f}'.format,
            'total_pnl': '{:,.2f}'.format
        }
    ))
    
    # Generate detailed reports for each top pair
    for pair, data in detailed_results.items():
        pair_folder = report_folder / pair
        pair_folder.mkdir(exist_ok=True)
        
        # Save detailed trade statistics
        data['trade_stats'].to_csv(pair_folder / 'trade_statistics.csv')
        data['performance_metrics'].to_csv(pair_folder / 'performance_metrics.csv')
        data['position_analysis'].to_csv(pair_folder / 'position_analysis.csv')
        
        # Generate performance visualization
        generate_performance_plots(pair, data, pair_folder)

def generate_performance_plots(pair: str, data: Dict[pd.DataFrame], output_folder: Path):
    """Generate comprehensive performance visualization for a pair"""
    # Returns and drawdown plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    data['performance_metrics']['cumulative_returns'].plot(label='Cumulative Returns')
    plt.title(f'Performance Analysis - {pair}')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    data['performance_metrics']['drawdown'].plot(label='Drawdown', color='red')
    plt.title('Drawdown Analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / 'returns_analysis.png')
    plt.close()
    
    # Trading activity and exposure plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    data['position_analysis']['zscore'].plot(label='Z-Score')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=2, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y=-2, color='red', linestyle='--', alpha=0.3)
    plt.title('Z-Score and Trading Signals')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    data['position_analysis'][['net_exposure', 'gross_exposure']].plot()
    plt.title('Position Exposure')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / 'trading_analysis.png')
    plt.close()


def save_detailed_results(results: List[PairTradingResult], output_folder: Path):
    """Save detailed trading results"""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_folder = output_folder / f'results_{timestamp}'
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Save main results DataFrame
    results_df = pd.DataFrame([{
        'pair': f"{r.pair[0]}-{r.pair[1]}",
        'start_date': r.start_date,
        'end_date': r.end_date,
        'metrics': r.metrics
    } for r in results])
    results_df.to_csv(results_folder / 'all_pairs_results.csv', index=False)
    
    # Create detailed analysis directory
    detailed_folder = results_folder / 'detailed_analysis'
    detailed_folder.mkdir(exist_ok=True)
    
    # Save detailed information for each pair
    for result in results:
        pair_name = f"{result.pair[0]}-{result.pair[1]}"
        pair_folder = detailed_folder / pair_name
        pair_folder.mkdir(exist_ok=True)
        
        # Save positions, returns, and exposures
        result.positions.to_csv(pair_folder / 'positions.csv')
        result.returns.to_csv(pair_folder / 'returns.csv')
        result.exposures.to_csv(pair_folder / 'exposures.csv')
        
        # Generate analysis plots
        generate_pair_analysis_plots(result, pair_folder)
    
    return results_folder, results_df

def format_top_pairs_output(results_df: pd.DataFrame) -> pd.DataFrame:
    """Format output for top pairs"""
    formatted_df = pd.DataFrame([{
        'Pair': f"{result['pair']}",
        'Calmar Ratio': result['metrics']['annual_return'] / abs(result['metrics']['max_drawdown']) if result['metrics']['max_drawdown'] != 0 else float('inf'),
        'Annual Return': result['metrics']['annual_return'],
        'Max Drawdown': result['metrics']['max_drawdown'],
        'Winning Trades': result['metrics']['winning_trades'],
        'Losing Trades': result['metrics']['losing_trades'],
        'Avg Win': result['metrics']['average_win'],
        'Avg Loss': result['metrics']['average_loss'],
        'Win Rate': result['metrics']['win_rate'],
        'Profit Factor': result['metrics']['profit_factor']
    } for _, result in results_df.iterrows()])
    
    return formatted_df


##############################################################################

def main():
    """Main execution function"""
    # Configuration
    DATA_FOLDER = Path('/Users/mouyasushi/k_data/永豐')
    OUTPUT_FOLDER = Path('/Users/mouyasushi/Desktop/pair_trading/output')
    CONFIG_PATH = 'config.json'
    START_DATE = '2022-10-14'
    END_DATE = '2024-10-14'
    
    try:
        # Initialize processor and strategy
        processor = PairsDataProcessor(DATA_FOLDER)
        strategy = PairsTradingStrategy(CONFIG_PATH)
        
        # Load and process data
        print("Loading and processing data...")
        all_stocks_daily = processor.combine_stock_data(START_DATE, END_DATE)
        
        # Execute strategy
        print("Executing pairs trading strategy...")
        results = []
        stock_codes = all_stocks_daily.columns
        pairs = [(stock1, stock2) 
                for i, stock1 in enumerate(stock_codes) 
                for stock2 in stock_codes[i+1:]]
        
        # Process pairs with progress bar
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
                        logging.error(f"Error with pair {pair}: {str(e)}")
                    pbar.update(1)
        
        if not results:
            logging.warning("No valid pairs found.")
            return
        
        # Save and analyze results
        results_folder, results_df = save_detailed_results(results, OUTPUT_FOLDER)
        
        # Perform additional analysis
        trading_frequency = analyze_trading_frequency(results_df)
        spread_analysis = analyze_spreads(results)
        trade_characteristics = analyze_trade_characteristics(results_df)
        
        # Save analysis results
        analysis_folder = results_folder / 'analysis'
        analysis_folder.mkdir(exist_ok=True)
        
        trading_frequency.to_csv(analysis_folder / 'trading_frequency.csv')
        spread_analysis.to_csv(analysis_folder / 'spread_analysis.csv')
        trade_characteristics.to_csv(analysis_folder / 'trade_characteristics.csv')
        
        # Display top 20 pairs
        print("\nTop 20 Pairs by Calmar Ratio:")
        print("=" * 120)
        formatted_df = format_top_pairs_output(results_df)
        top_20_pairs = formatted_df.nlargest(20, 'Calmar Ratio')
        print(top_20_pairs.to_string(
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
        
        print(f"\nAnalysis completed. Results saved to {results_folder}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    
    summary_df, detailed_results = analyze_best_pairs(results)
    
    # Generate comprehensive report
    generate_performance_report(summary_df, detailed_results, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()