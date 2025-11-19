"""stat_tests.py

Simple script to perform a few statistical tests on columns read from
a whitespace-separated text file. The input file is expected to have a
header row (column names) followed by numeric rows. Decimal commas are
accepted and converted to dots.

Usage (from project root):
    python stat_tests/stat_tests.py --input stat_tests/results.txt \
        --output stat_tests_v2.txt

The script writes Shapiro-Wilk normality results, Levene homogeneity
results (pairwise), and a series of t-tests (using column index 5 vs
all columns) into the output file.
"""

from scipy import stats
import pandas as pd
import argparse
from pathlib import Path


def read_numeric_table(path: Path):
    """Read a whitespace-separated table with header from `path`.

    Converts comma decimal separators to dots and returns a pandas
    DataFrame with float values.
    """
    with open(path, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()

    # Header is the first line, values thereafter
    header = lines[0].strip().split()

    float_data = []
    for line in lines[1:]:
        values = line.strip().split()
        tmp_list = []
        for value in values:
            # Accept numbers with comma as decimal separator
            value = value.replace(',', '.')
            tmp_list.append(float(value))
        float_data.append(tmp_list)

    df = pd.DataFrame(float_data, columns=header)
    return df


def run_tests(df: pd.DataFrame, out_path: Path):
    """Run statistical tests on the DataFrame `df` and write results.

    - Shapiro-Wilk for each column (normality)
    - Levene's test pairwise (homogeneity)
    - Independent t-tests comparing column index 5 to all columns
    """
    populations = len(df.columns)
    with open(out_path, 'w', encoding='utf-8') as out_f:
        # Shapiro-Wilk normality test for each population
        for i in range(populations):
            population = df.iloc[:, i].tolist()
            shapiro_result = stats.shapiro(population)
            out_f.write(f"Shapiro-Wilk test for normality for population {df.columns[i]}: {shapiro_result}\n")

        # Pairwise Levene test for homogeneity of variances
        for i in range(populations):
            population1 = df.iloc[:, i].tolist()
            for j in range(i + 1, populations):
                population2 = df.iloc[:, j].tolist()
                levene_result = stats.levene(population1, population2)
                out_f.write(f"Homogeneity test for population {df.columns[i]} and {df.columns[j]}: {levene_result}\n")

        # T-tests comparing column index 5 against all columns (keeps original logic)
        # Note: this assumes there is a column at index 5 — will raise IndexError otherwise.
        population1 = df.iloc[:, 5].tolist()
        for j in range(0, populations):
            population2 = df.iloc[:, j].tolist()
            ttest_result = stats.ttest_ind(population1, population2)
            out_f.write(f"T-test population {df.columns[5]} and {df.columns[j]}: {ttest_result}\n")


def main():
    parser = argparse.ArgumentParser(description='Run basic statistical tests on a table file.')
    parser.add_argument('--input', '-i', required=False, default='stat_tests/results.txt',
                        help='Path to input text file (header + whitespace-separated numeric rows)')
    parser.add_argument('--output', '-o', required=False, default='stat_tests_v2.txt',
                        help='Path to output results file')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = read_numeric_table(input_path)
    run_tests(df, output_path)


if __name__ == '__main__':
    main()
