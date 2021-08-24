import os
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

parser = ArgumentParser(
    description="ConveRT model trainer",
    formatter_class=RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--input_data_dir",
    required=True,
    help="directory with input data files",
)
args = parser.parse_args()

print(args.input_data_dir)
print(os.listdir(args.input_data_dir))
