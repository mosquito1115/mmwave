import os
import argparse


def count_files_in_dir(path: str) -> int:
    try:
        return sum(1 for entry in os.scandir(path) if entry.is_file())
    except FileNotFoundError:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check each immediate subdirectory and print those not having the expected number of files.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory)",
    )
    parser.add_argument(
        "-n",
        "--expected",
        type=int,
        default=11,
        help="Expected number of files in each subdirectory (default: 11)",
    )

    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        raise SystemExit(1)

    mismatched = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_dir():
                    file_count = count_files_in_dir(entry.path)
                    if file_count != args.expected:
                        mismatched.append((entry.name, file_count))
    except PermissionError:
        print(f"Permission denied when scanning: {root}")
        raise SystemExit(1)

    for name, count in mismatched:
        print(name)

    # Exit code non-zero if any mismatches found (useful for CI)
    raise SystemExit(0 if not mismatched else 2)


if __name__ == "__main__":
    # example: python .\check_subdir_counts.py "E:\high_DiskSpy\1029" -n 11
    main()

