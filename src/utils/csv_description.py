import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='''
Compute and store a description (including mean and std) of a csv that will
later be used by the environment to normalize the data.
''')

parser.add_argument('file')
args = parser.parse_args()


def main():
    df = pd.read_csv(args.file, sep=';')
    description = df.describe()

    output_path = f'{args.file[:-4]}_description.csv'
    description.to_csv(output_path, sep=';')


if __name__ == '__main__':
    main()
