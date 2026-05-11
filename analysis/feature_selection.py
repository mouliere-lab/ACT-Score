# Author: Parisa Mapar

import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add repository root to Python path so modules in src/ can be imported.
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.elastic_net_feature_selection import run_elastic_net_feature_selection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Elastic Net feature selection for a selected feature group."
    )

    parser.add_argument(
        "--feature_group",
        type=str,
        required=True,
        choices=["LIONHEART", "DELFI"],
        help="Feature group name used as the output prefix.",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input CSV file containing features and metadata.",
    )

    parser.add_argument(
        "--feature_file",
        type=str,
        required=True,
        help="CSV file containing feature names.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the feature_selection output folder will be created.",
    )

    parser.add_argument(
        "--min_times_selected",
        type=int,
        default=5,
        help="Minimum number of random-seed runs required to keep a feature.",
    )

    parser.add_argument(
        "--random_states",
        type=int,
        default=10,
        help="Number of random seeds.",
    )

    parser.add_argument(
        "--timepoint",
        type=int,
        default=1,
        help="Timepoint used for feature selection.",
    )

    parser.add_argument(
        "--outcome_col",
        type=str,
        default="2Y_PFS",
        help="Outcome column name.",
    )

    parser.add_argument(
        "--feature_col",
        type=str,
        default="feature",
        help="Column containing feature names in feature_file.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=16,
        help="Number of parallel jobs.",
    )

    args = parser.parse_args()

    run_elastic_net_feature_selection(
        input_file=args.input_file,
        feature_file=args.feature_file,
        output_dir=args.output_dir,
        output_prefix=args.feature_group,
        min_times_selected=args.min_times_selected,
        random_states=args.random_states,
        timepoint=args.timepoint,
        outcome_col=args.outcome_col,
        feature_col=args.feature_col,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()


# Example for LIONHEART:
# python analysis/feature_selection.py \
#   --feature_group LIONHEART \
#   --input_file data/example_input.csv \
#   --feature_file data/feature_lists/LIONHEART_features.csv \
#   --output_dir results \
#   --min_times_selected 5 \
#   --random_states 10 \
#   --timepoint 1 \
#   --outcome_col 2Y_PFS \
#   --feature_col feature \
#   --n_jobs 16
#
# Example for DELFI:
# python analysis/feature_selection.py \
#   --feature_group DELFI-FTK \
#   --input_file data/example_input.csv \
#   --feature_file data/feature_lists/DELFI-FTK_features.csv \
#   --output_dir results \
#   --min_times_selected 5 \
#   --random_states 10 \
#   --timepoint 1 \
#   --outcome_col 2Y_PFS \
#   --feature_col feature \
#   --n_jobs 16
