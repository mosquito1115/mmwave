import os
import csv
from typing import Optional, Tuple


def parse_times_from_name(name: str) -> Optional[Tuple[float, float, int, int]]:
    """
    Parse time_vib and time_still from a directory name following a pattern like:
    "..._disk_8_75_25_v9.mat" -> time_vib=0.075, time_still=0.025

    Logic:
    - Split by '_' and locate the token 'disk'.
    - Collect numeric tokens immediately after 'disk'.
    - Interpret the 2nd and 3rd numeric tokens as vib_ms and still_ms.
    - Convert to seconds by dividing by 1000.

    Returns (time_vib_s, time_still_s, vib_ms, still_ms), or None if not parseable.
    """
    base = os.path.splitext(name)[0]
    tokens = base.split('_')
    try:
        idx = tokens.index('disk')
    except ValueError:
        return None

    nums_after = []
    for tok in tokens[idx + 1:]:
        # Stop if token starts with a non-digit and contains letters (e.g., 'v9')
        if tok.isdigit():
            nums_after.append(int(tok))
        else:
            # strip leading non-digits and trailing non-digits, then check digits-only
            stripped = ''.join(ch for ch in tok if ch.isdigit())
            if stripped.isdigit():
                # Only accept if original token starts with a digit (avoid 'v9')
                if tok[0].isdigit():
                    nums_after.append(int(stripped))
                elif tok[0] == 'v':
                    number=int(tok[1:])
                elif tok[0:4] == 'test':
                    number=int(tok[4:])
                else:
                    break
            else:
                break

    if len(nums_after) < 3:
        return None

    # nums_after[0] is typically an extra parameter (e.g., distance/speed). We ignore it.
    vib_ms = nums_after[1]
    still_ms = nums_after[2]
    return vib_ms / 1000.0, still_ms / 1000.0, number


def main():
    # Configure parameters here (no CLI parsing)
    # Example: root directory containing subdirectories to scan
    root = "E:\\high_DiskSpy\\1114\\disk9\\saveData\\"
    # Optional: set a specific CSV output path; leave None to use ROOT/subdir_times.csv
    output = None

    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        raise SystemExit(1)

    out_path = output or os.path.join("E:\\high_DiskSpy\\csv", 'disk_9.csv')

    rows = []
    with os.scandir(root) as it:
        for entry in it:
            name = entry.name
            parsed = parse_times_from_name(name)
            if parsed is None:
                rows.append({
                    'number' : '',
                    'subdir': name,
                    'time_vib_s': '',
                    'time_still_s': '',
                    'pattern': '101010',
                    'path': entry.path,
                })
            else:
                tv_s, ts_s, number = parsed
                rows.append({
                    'number': number,
                    'subdir': name,
                    'time_vib_s': f"{tv_s:.6f}",
                    'time_still_s': f"{ts_s:.6f}",
                    'pattern': '101010',
                    'path': entry.path,
                })

    # Write CSV (utf-8-sig for better Excel compatibility on Windows)
    fieldnames = ['number','subdir', 'time_vib_s', 'time_still_s', 'pattern', 'path']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(1 for r in rows)
    print(f"Wrote {total} rows to: {out_path}")


if __name__ == '__main__':
    main()
