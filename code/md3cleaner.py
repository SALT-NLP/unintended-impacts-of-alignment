import argparse
import os
import pandas as pd


def clean_md3_files(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.startswith("transcripts") and filename.endswith(".tsv"):
            print(filename)
            dialect = filename.split("transcripts_")[1][0:-4]
            print(f"Processing {dialect} dialect")

            dialect_dir = os.path.join(output_dir, dialect)
            if not os.path.exists(dialect_dir):
                os.mkdir(dialect_dir)

            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path, delimiter='\t')
            df.drop_duplicates(inplace=True)  # Drop duplicate rows
            
            grouped_df = df.groupby('clip_identifier')
            
            for clip_identifier, group in grouped_df:

                outfile = clip_identifier + ".txt"
                file_path = os.path.join(dialect_dir, outfile)
                with open(file_path, 'w') as file:
                    previous_transcript = {}
                    for _, row in group.iterrows():
                        speaker_id = row['speaker_id']
                        transcript = row['transcript']
                        
                        if isinstance(speaker_id, float) or isinstance(transcript, float):
                            continue
                        
                        if speaker_id in previous_transcript and previous_transcript[speaker_id] == transcript:
                            continue
                        
                        previous_transcript[speaker_id] = transcript
                        
                        file.write(f"{speaker_id.strip()}: {transcript.strip()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MD3 Cleaner")
    parser.add_argument("--input_dir", help="Input directory path")
    parser.add_argument("--output_dir", help="Output directory path")
    args = parser.parse_args()

    clean_md3_files(args.input_dir, args.output_dir)
