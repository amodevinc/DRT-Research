import pandas as pd
import argparse
import os
from tabulate import tabulate

def read_parquet_file(file_path, output_format='table', sample_rows=None, save_output=None):
    """
    Read a Parquet file and display it in a human-readable format.
    
    Parameters:
    - file_path: Path to the Parquet file
    - output_format: Format to display ('table', 'records', or 'csv')
    - sample_rows: Number of rows to display (None for all)
    - save_output: Path to save the output (None for no saving)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    try:
        # Read the Parquet file
        df = pd.read_parquet(file_path)
        
        # Take a sample if specified
        if sample_rows is not None and sample_rows > 0:
            if sample_rows < len(df):
                df = df.head(sample_rows)
            print(f"Displaying {len(df)} rows of {file_path}")
        else:
            print(f"Displaying all {len(df)} rows of {file_path}")
        
        # Print shape and data types
        print(f"\nDataFrame Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\nData Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Display the data based on the specified format
        print("\nData Preview:")
        if output_format == 'table':
            # Display as a nicely formatted table
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=True))
        elif output_format == 'records':
            # Display as records (list of dictionaries)
            records = df.to_dict('records')
            for i, record in enumerate(records):
                print(f"\nRecord {i+1}:")
                for key, value in record.items():
                    print(f"  {key}: {value}")
        elif output_format == 'csv':
            # Display as CSV
            print(df.to_csv(index=False))
        
        # Save output if specified
        if save_output:
            if save_output.endswith('.csv'):
                df.to_csv(save_output, index=False)
                print(f"\nData saved to {save_output}")
            elif save_output.endswith('.xlsx'):
                df.to_excel(save_output, index=False)
                print(f"\nData saved to {save_output}")
            elif save_output.endswith('.json'):
                df.to_json(save_output, orient='records')
                print(f"\nData saved to {save_output}")
            else:
                print(f"\nUnsupported output format. Please use .csv, .xlsx, or .json extension.")
        
        return df
    
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Read a Parquet file and display it in human-readable format.')
    parser.add_argument('file_path', help='Path to the Parquet file')
    parser.add_argument('--format', choices=['table', 'records', 'csv'], default='csv',
                        help='Output format (table, records, or csv)')
    parser.add_argument('--rows', type=int, default=None, 
                        help='Number of rows to display (default: all rows)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the output (supports .csv, .xlsx, .json)')
    
    args = parser.parse_args()
    
    # Read and display the Parquet file
    read_parquet_file(args.file_path, args.format, args.rows, args.save)