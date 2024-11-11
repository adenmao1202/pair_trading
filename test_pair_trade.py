# test_pairs_trading.py
import pandas as pd
from pathlib import Path
from basic_coint import PairsDataProcessor, PairsTradingStrategy
import concurrent.futures
from datetime import datetime
from tqdm import tqdm

def test_with_sample_data():
    """
    Test the pairs trading strategy with a small sample of real data
    """
    # Configuration
    DATA_FOLDER = Path('/Users/mouyasushi/k_data/永豐')
    OUTPUT_FOLDER = Path('./test_output')
    START_DATE = '2023-01-14'  
    END_DATE = '2024-10-01'
    
    # Create test output directory
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Test Data Loading
        print("1. Testing Data Loading...")
        processor = PairsDataProcessor(DATA_FOLDER)
        
        # Load first 5 CSV files only for testing
        csv_files = list(DATA_FOLDER.glob('*.csv'))[:5]   
        print(f"Testing with {len(csv_files)} stock files")
        
        # Test individual file loading
        print("\nTesting individual file loading:")
        for csv_file in csv_files[:2]:  
            stock_df = processor.load_stock_data(csv_file.stem, csv_file)
            print(f"Loaded {csv_file.stem}: Shape {stock_df.shape if stock_df is not None else 'None'}")
        
        # 2. Test Data Combining
        print("\n2. Testing Data Combining...")
        all_stocks_daily = processor.combine_stock_data(START_DATE, END_DATE)
        print(f"Combined data shape: {all_stocks_daily.shape}")
        print("Sample of combined data:")
        print(all_stocks_daily.head())
        
        # 3. Test Strategy Initialization
        print("\n3. Testing Strategy Setup...")
        strategy = PairsTradingStrategy(
            lookback_period=20,
            enter_long_zscore_threshold=1.0,
            enter_short_zscore_threshold=1.0,
            exit_zscore_threshold=0.0
        )
        
        # 4. Test Pair Trading Execution
        print("\n4. Testing Pair Trading Execution...")
        stock_codes = all_stocks_daily.columns[:]  
        results = []
        
        # Generate test pairs
        test_pairs = []
        for i, stock1 in enumerate(stock_codes):
            for stock2 in stock_codes[i+1:]:
                test_pairs.append((stock1, stock2))
        
        print(f"Testing with {len(test_pairs)} pairs")
        
        # Execute strategy for each pair
        for pair in tqdm(test_pairs, desc="Processing pairs"):
            try:
                result = strategy.execute_pair_trade(
                    all_stocks_daily[pair[0]],
                    all_stocks_daily[pair[1]],
                    pair
                )
                if result is not None:
                    results.append(result)
                    print(f"\nProcessed pair {pair}:")
                    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
                    print(f"Winning Trades: {result.metrics['winning_trades']}")
                    print(f"Losing Trades: {result.metrics['losing_trades']}")
                    print(f"Average Win: {result.metrics['average_win']:.2%}")
                    print(f"Average Loss: {result.metrics['average_loss']:.2%}")
                    print(f"Win Rate: {result.metrics['win_rate']:.2%}")
                    print(f"Annual Return: {result.metrics['annual_return']:.2%}")
            except Exception as e:
                print(f"Error processing pair {pair}: {str(e)}")
        
        # 5. Test Results Processing
        print("\n5. Testing Results Processing...")
        if results:
            # Create results DataFrame
            results_df = pd.DataFrame([{
                'pair': f"{r.pair[0]}-{r.pair[1]}",
                'start_date': r.start_date,
                'end_date': r.end_date,
                **r.metrics
            } for r in results])
            
            # Save test results with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_df.to_csv(OUTPUT_FOLDER / f'test_results_{timestamp}.csv', index=False)
            
            # Calculate Calmar Ratio with safety checks
            def safe_calmar_ratio(row):
                try:
                    if row['max_drawdown'] == 0:
                        return float('inf')
                    return abs(row['annual_return']) / abs(row['max_drawdown'])
                except:
                    return float('nan')

            results_df['calmar_ratio'] = results_df.apply(safe_calmar_ratio, axis=1)
            
            # Sort by Calmar Ratio and get top 10
            top_10_pairs = results_df.nlargest(10, 'calmar_ratio')
            
            # Print summary statistics
            print(f"\nTotal pairs tested: {len(test_pairs)}")
            print(f"Valid pairs found: {len(results_df)}")
            print(f"Success rate: {len(results_df)/len(test_pairs):.2%}")
            
            # Print formatted summary of top 10 pairs
            print("\nTop 10 Pairs by Calmar Ratio:")
            print("=" * 120)
            formatted_df = top_10_pairs[[
                'pair',
                'calmar_ratio',
                'annual_return',
                'max_drawdown',
                'winning_trades',
                'losing_trades',
                'average_win',
                'average_loss',
                'win_rate'
            ]].copy()
            
            formatted_df.columns = [
                'Pair',
                'Calmar Ratio',
                'Annual Return',
                'Max Drawdown',
                'Winning Trades',
                'Losing Trades',
                'Avg Win',
                'Avg Loss',
                'Win Rate'
            ]
            
            # Format the output with alignment
            pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x) if abs(x) >= 1 else '{:.4f}'.format(x))
            print(formatted_df.to_string(
                formatters={
                    'Calmar Ratio': '{:>8.2f}'.format,
                    'Annual Return': '{:>8.2%}'.format,
                    'Max Drawdown': '{:>8.2%}'.format,
                    'Avg Win': '{:>8.2%}'.format,
                    'Avg Loss': '{:>8.2%}'.format,
                    'Win Rate': '{:>8.2%}'.format
                }
            ))
            print("=" * 120)
            
            # Save detailed results for top pairs
            top_pairs_folder = OUTPUT_FOLDER / 'top_pairs'
            top_pairs_folder.mkdir(parents=True, exist_ok=True)
            
            # Save top 10 results to a separate CSV with timestamp
            top_10_pairs.to_csv(top_pairs_folder / f'top_10_pairs_{timestamp}.csv', index=False)
            
            # Print detailed statistics for the best pair
            best_pair = top_10_pairs.iloc[0]
            print("\nBest Pair Details:")
            print("=" * 50)
            print(f"Pair: {best_pair['pair']}")
            print(f"Trading Period: {best_pair['start_date']} to {best_pair['end_date']}")
            print(f"\nPerformance Metrics:")
            print(f"  - Calmar Ratio: {best_pair['calmar_ratio']:.2f}")
            print(f"  - Annual Return: {best_pair['annual_return']:.2%}")
            print(f"  - Max Drawdown: {best_pair['max_drawdown']:.2%}")
            print(f"\nTrading Statistics:")
            print(f"  - Win Rate: {best_pair['win_rate']:.2%}")
            print(f"  - Total Trades: {best_pair['winning_trades'] + best_pair['losing_trades']}")
            print(f"  - Winning Trades: {best_pair['winning_trades']}")
            print(f"  - Losing Trades: {best_pair['losing_trades']}")
            print(f"\nAverage Returns:")
            print(f"  - Average Win: {best_pair['average_win']:.2%}")
            print(f"  - Average Loss: {best_pair['average_loss']:.2%}")
            print("=" * 50)
            
            # Save detailed results for the best pair
            best_pair_folder = top_pairs_folder / f"{best_pair['pair']}_{timestamp}"
            best_pair_folder.mkdir(parents=True, exist_ok=True)
            
            # Find and save the detailed results for the best pair
            for result in results:
                if f"{result.pair[0]}-{result.pair[1]}" == best_pair['pair']:
                    result.positions.to_csv(best_pair_folder / 'positions.csv')
                    pd.Series(result.returns).to_csv(best_pair_folder / 'returns.csv')
                    result.exposures.to_csv(best_pair_folder / 'exposures.csv')
                    break
                    
        else:
            print("No valid pairs found in test data")
        
    except Exception as e:
        print(f"Error in test execution: {str(e)}")
        raise
    
    print("\nTest execution completed!")

if __name__ == "__main__":
    test_with_sample_data()