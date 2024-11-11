from pathlib import Path
import concurrent.futures
import pandas as pd
from basic_coint import PairsDataProcessor, PairsTradingStrategy

def main():
    # Configuration
    DATA_FOLDER = Path('/Users/mouyasushi/k_data/永豐')
    OUTPUT_FOLDER = Path('/Users/mouyasushi/Desktop/pair_trading/output')
    START_DATE = '2022-10-14'
    END_DATE = '2024-10-14'
    
    # Create output directory if it doesn't exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor and strategy
    processor = PairsDataProcessor(DATA_FOLDER)
    strategy = PairsTradingStrategy()
    
    try:
        # Load and process data
        print("Loading and processing data...")
        all_stocks_daily = processor.combine_stock_data(START_DATE, END_DATE)
        
        # Execute strategy
        print("Executing pairs trading strategy...")
        results = []
        stock_codes = all_stocks_daily.columns
        total_pairs = len(stock_codes) * (len(stock_codes) - 1) // 2
        processed_pairs = 0
        
        # Batch processing setup
        batch_size = 100
        pairs = []
        
        # Generate all pair combinations
        for i, stock1 in enumerate(stock_codes):
            for stock2 in stock_codes[i+1:]:
                pairs.append((stock1, stock2))
        
        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
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
                            
                        if processed_pairs % 10 == 0:
                            progress = (processed_pairs / total_pairs) * 100
                            print(f"Progress: {progress:.1f}% ({processed_pairs}/{total_pairs} pairs)")
                            
                    except Exception as e:
                        print(f"Error with pair {pair}: {str(e)}")
        
        print(f"\nProcessing complete. Found {len(results)} valid pairs out of {total_pairs} total pairs.")
        
        if not results:
            print("No valid pairs found.")
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame([{
            'pair': f"{r.pair[0]}-{r.pair[1]}",
            'start_date': r.start_date,
            'end_date': r.end_date,
            **r.metrics
        } for r in results])
        
        # Save detailed results
        save_detailed_results(results, results_df, OUTPUT_FOLDER)
        
        # Display best pairs
        display_best_pairs(results_df)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

def save_detailed_results(results, results_df, output_folder):
    """Save detailed results and analysis"""
    # Save main results
    results_df.to_csv(output_folder / 'pairs_trading_results.csv', index=False)
    
    # Save detailed analysis for each pair
    for result in results:
        pair_name = f"{result.pair[0]}-{result.pair[1]}"
        pair_folder = output_folder / 'pair_details' / pair_name
        pair_folder.mkdir(parents=True, exist_ok=True)
        
        # Save positions and signals
        result.positions.to_csv(pair_folder / 'positions.csv')
        
        # Save returns
        pd.Series(result.returns).to_csv(pair_folder / 'returns.csv')
        
        # Save exposures
        result.exposures.to_csv(pair_folder / 'exposures.csv')
        
        # Save metrics
        pd.Series(result.metrics).to_frame().to_csv(pair_folder / 'metrics.csv')

def display_best_pairs(results_df):
    """Display best performing pairs by different metrics"""
    print("\nTop 5 Pairs by Sharpe Ratio:")
    print(results_df.nlargest(5, 'sharpe_ratio')[['pair', 'sharpe_ratio', 'total_return', 'win_rate']])
    
    print("\nTop 5 Pairs by Total Return:")
    print(results_df.nlargest(5, 'total_return')[['pair', 'total_return', 'sharpe_ratio', 'win_rate']])
    
    print("\nTop 5 Pairs by Win Rate:")
    print(results_df.nlargest(5, 'win_rate')[['pair', 'win_rate', 'total_return', 'sharpe_ratio']])
    
    print("\nTop 5 Pairs by Profit Factor:")
    print(results_df.nlargest(5, 'profit_factor')[['pair', 'profit_factor', 'total_return', 'sharpe_ratio']])

if __name__ == "__main__":
    main()