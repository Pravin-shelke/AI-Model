import pandas as pd
import json
import sys

# Usage: python convert_mock_to_training.py mock_test_data.csv output_training_format.csv


def extract_indicators_and_labels(df):
    """
    Scan all rows to collect all unique soaId (indicator codes) and their labelNames.
    Returns: (set of soaIds, dict of soaId -> labelName)
    """
    indicators = set()
    label_map = {}
    for row in df['labelsAnswersMap']:
        try:
            items = json.loads(row)
            for item in items:
                if 'soaId' in item and item['soaId']:
                    indicators.add(item['soaId'])
                    if item.get('labelName'):
                        label_map[item['soaId']] = item['labelName']
        except Exception:
            continue
    return indicators, label_map

def expand_row(row, indicators):
    """
    For a single row, return a dict of {soaId: answer} for all indicators.
    """
    answers = {soaId: None for soaId in indicators}
    try:
        items = json.loads(row['labelsAnswersMap'])
        for item in items:
            if 'soaId' in item and item['soaId']:
                answers[item['soaId']] = item.get('answer')
    except Exception:
        pass
    return answers


def main(input_csv, output_csv, label_map_csv):
    df = pd.read_csv(input_csv, dtype=str)
    print(f"Loaded {len(df)} records from {input_csv}")

    indicators, label_map = extract_indicators_and_labels(df)
    indicators = sorted(indicators)
    print(f"Found {len(indicators)} unique indicators (soaId)")

    # Expand each row
    expanded = df.apply(lambda row: expand_row(row, indicators), axis=1)
    expanded_df = pd.DataFrame(list(expanded))

    # Merge with metadata columns (all except labelsAnswersMap)
    meta_cols = [col for col in df.columns if col != 'labelsAnswersMap']
    result = pd.concat([df[meta_cols].reset_index(drop=True), expanded_df.reset_index(drop=True)], axis=1)

    # Save to output
    result.to_csv(output_csv, index=False)
    print(f"Saved expanded training data to {output_csv}")

    # Save label map for reference (soaId,labelName)

    with open(label_map_csv, 'w', encoding='utf-8') as f:
        f.write('soaId,labelName\n')
        for soaId in indicators:
            label = label_map.get(soaId, '')
            f.write(f'"{soaId}","{label}"\n')
    print(f"Saved indicator label map to {label_map_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_mock_to_training.py mock_test_data.csv output_training_format.csv indicator_labels.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
