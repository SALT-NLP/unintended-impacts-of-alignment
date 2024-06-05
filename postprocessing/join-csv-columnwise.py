import pandas as pd
import os
import argparse

# Define the directory to search for CSV files
directory = '/path/to/directory'

# Define the output file path
output_file = '/path/to/output.csv'

def parse_args():
    parser = argparse.ArgumentParser(description="Join CSV files in a directory.")
    parser.add_argument(
        "--directory",
        type=str,
        default=directory,
        help="The directory to search for CSV files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=output_file,
        help="The output file path",
    )
    args = parser.parse_args()
    return args

def join_csv_columnwise(directory, output_file):
    # List of dataframes to join
    dataframes = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                # Read the CSV file
                df = pd.read_csv(os.path.join(root, file))
                # Append the dataframe to the list
                dataframes.append(df)

    # Initialize the new dataframe with the first dataframe
    new_df = dataframes[0]

    # Iterate over the remaining dataframes
    for df in dataframes[1:]:
        # Find shared columns
        shared_cols = list(set(new_df.columns) & set(df.columns))
        
        if shared_cols:
            # Join on shared columns
            new_df = pd.merge(new_df, df, on=shared_cols)
        else:
            # Add different columns to the new dataframe
            new_cols = list(set(df.columns) - set(new_df.columns))
            for col in new_cols:
                new_df[col] = df[col]

    # Create the output directory if it does not exist
    if os.path.dirname(output_file) != "" and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Write the joined data to the output CSV file
    new_df.to_csv(output_file, index=False)


def main():
    args = parse_args()

    join_csv_columnwise(args.directory, args.output_file)

if __name__ == '__main__':
    main()

# Example usage:
# python join-csv-columnwise.py --directory /path/to/directory --output_file /path/to/output.csv
# python postprocessing/join-csv-columnwise.py --directory outputs/pubmedqa/ --output_file results/7_pubmedqa/pubmedqa_generations.csv
# python postprocessing/join-csv-columnwise.py --directory outputs/belebele/ --output_file results/8_belebele/belebele.csv