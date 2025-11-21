import os
import csv
from typing import List, Tuple, Optional

# We rely on functions defined in split.py
from detect_packet_header import detect_packet_header

def main():
    # Configure parameters here (no CLI)
    root = "E:\\high_DiskSpy\\csv\\"
    input_csv = os.path.join(root, 'disk_9.csv')
    output_csv = os.path.join(root, 'disk_9_results_with_unwrap.csv')

    rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = []
    for row in rows:
        number = (row.get('number') or '').strip()
        subdir = (row.get('subdir') or '').strip()
        path = (row.get('path') or '').strip()
        pattern_str = (row.get('pattern') or '').strip()
        time_vib_s = float((row.get('time_vib_s') or '').strip())
        time_still_s = float((row.get('time_still_s') or '').strip())

        for sample_idx in (19,20):
            for antenna_idx in (0, 1, 2, 3, 11, 12):
                out_row = {
                    'number': number,
                    'subdir': subdir,
                    'time_vib_s': time_vib_s,
                    'time_still_s': time_still_s,
                    'pattern': pattern_str,
                    'sample_idx': sample_idx,
                    'antenna_idx': antenna_idx,
                    'start_sample': '',
                    'path': path,
                }

                start_sample=detect_packet_header(
                    mat_path=path,
                    pattern_str=pattern_str,
                    time_vib=time_vib_s,
                    time_still=time_still_s,
                    antenna_idx=antenna_idx,
                    sample_idx=sample_idx,
                )
                out_row['start_sample'] = start_sample

                results.append(out_row)

    fieldnames = [
        'number', 'subdir', 'time_vib_s', 'time_still_s', 'pattern', 'sample_idx', 'antenna_idx', 'start_sample', 'path'
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    print(f"Wrote {total} rows to: {output_csv}")


if __name__ == '__main__':
    main()
