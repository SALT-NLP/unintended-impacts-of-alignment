import os
import csv
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
    parser.add_argument(
        "--belebele",
        action='store_true',
        default=False,
        help="Whether or not to handle the belebele csv output format",
    )
    parser.add_argument(
        "--drop_columns",
        type=str,
        default=None,
        help="The columns to drop from the csv",
    )
    args = parser.parse_args()
    return args

def join_csv(directory, output_file, belebele=False, drop_columns=None):
    """
    Join all CSV files in a directory and its subdirectories into a single CSV file.
    """
    # Initialize an empty list to store the data from all CSV files
    all_data = []

    # Initialize an empty list to store the header of each CSV file
    header = []

    # Loop through directories and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                if file.endswith('.csv'):
                    # Read the CSV file
                    with open(os.path.join(root, file), 'r') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        # Get the header row
                        new_header = next(csv_reader)
                        if len(header) < len(new_header):
                            header = new_header

                        # Append the data from the CSV file to the list
                        all_data.extend(csv_reader)

    if belebele:
        model_name = header[3][:-6]
        for i in range(len(new_header)):
            header[i] = header[i].replace(model_name, '')

    keep_indices = list(range(len(header)))
    if drop_columns:
        keep_indices = [i for i, col in enumerate(header) if col not in drop_columns.split(',')]
        header = [header[i] for i in keep_indices]

    if not all_data:
        raise ValueError('No CSV files found in the directory.')
    
    # Create the output directory if it does not exist
    if os.path.dirname(output_file) != "" and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Write the joined data to the output CSV file
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(header)

        # Write the data rows
        for row in all_data:
            if len(row) < len(header):
                row.extend([False] * (len(header) - len(row)))
            row = [row[i] for i in keep_indices]
            csv_writer.writerow(row)

def main():
    args = parse_args()

    join_csv(args.directory, args.output_file, args.belebele, args.drop_columns)

if __name__ == '__main__':
    main()

# Example usage:
# python postprocessing/join-csv.py --directory /path/to/directory --output_file /path/to/output.csv

# python postprocessing/join-csv.py --directory  ./outputs/tydiqa-goldp-nll/ --output_file ./results/4_tydiqa/goldp-nll.csv

# python postprocessing/join-csv.py --directory  ./outputs/globalopinionsqa --output_file ./results/5_globalopinionsqa/globalopinionsqa.csv

# python postprocessing/join-csv.py --directory  ./outputs/md3-game/ --output_file ./results/6_md3game/md3game.csv
    
# python postprocessing/join-csv.py --directory  ./outputs/belebele/ --output_file ./results/8_belebele/belebele.csv --belebele --drop_columns flores_passage,question,prob_A,prob_B,prob_C,prob_D,,mc_answer1,mc_answer2,mc_answer3,mc_answer4,link,question_num