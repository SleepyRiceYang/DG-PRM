import argparse
import os
from pathlib import Path

from src.visualizer import Visualizer
from src.analyzers.mode_deep_analyzer import ModeDeepAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Mode Deep Trajectory Analysis Tool")
    parser.add_argument("--base", type=str, required=True,
                        help="Base directory containing the JSON result files")
    parser.add_argument("--files", type=str, nargs='+', required=True,
                        help="JSON filenames (relative to base) to analyze and compare")
    parser.add_argument("--output", type=str, default="analysis_v1",
                        help="Output directory name (created under base)")
    parser.add_argument("--auto_name", action="store_true", default=True,
                        help="Automatically format mode names from filenames")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base)
    output_root = base_dir / args.output

    # For deep analysis, we use the specific sub-folder for its plots
    # ModeDeepAnalyzer handles its own sub-folder structure for reports
    viz = Visualizer(str(output_root))

    analyzer = ModeDeepAnalyzer(
        base_dir=str(base_dir),
        output_name=args.output,
        visualizer=viz
    )

    for fname in args.files:
        added = analyzer.add_file(fname, auto_name=args.auto_name)
        if added:
            print(f"[*] Registered: {fname}")
        else:
            print(f"[!] Failed to register: {fname}")

    analyzer.run()


if __name__ == "__main__":
    main()
