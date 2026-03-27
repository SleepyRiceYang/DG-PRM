import argparse
import os

from src.visualizer import Visualizer
from src.analyzers.unique_segment_analyzer import UniqueSegmentAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Unique-Segment Rollout Analyzer (single mode)")
    parser.add_argument("--rollout_file", type=str, required=True,
                        help="Path to a single rollout results JSON file")
    parser.add_argument("--mode", type=str, required=True,
                        help="Mode name tag (e.g., dissonance/random)")
    parser.add_argument("--remark", type=str, default="unique_v1",
                        help="Analysis remark tag (will appear in meta)")
    parser.add_argument("--output_name", type=str, default="analysis_results",
                        help="Output directory name (created under rollout file's directory)")
    return parser.parse_args()


def main():
    args = parse_args()

    rollout_path = os.path.abspath(args.rollout_file)
    base_dir = os.path.dirname(rollout_path)
    output_dir = os.path.join(base_dir, args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    viz = Visualizer(output_dir)
    analyzer = UniqueSegmentAnalyzer(
        rollout_path=rollout_path,
        output_dir=output_dir,
        mode_name=args.mode,
        remark=args.remark,
        visualizer=viz,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
