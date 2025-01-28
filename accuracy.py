def calculate_accuracy_from_json(json_file_path):
    """
    Calculate accuracy from JSON results file

    Args:
        json_file_path (str): Path to the JSON file containing results

    Returns:
        float: Accuracy score between 0 and 1
    """
    import json
    import pandas as pd
    from pathlib import Path

    # Read the results JSON file
    try:
        with open(json_file_path, 'r') as f:
            results = json.load(f)
            results = pd.DataFrame(results)
    except FileNotFoundError:
        print(f"Error: Could not find results file at {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return None

    # Try to find the test data file
    test_file_paths = [
        'data/snli_1.0/snli_1.0_test.jsonl',
        '../data/snli_1.0/snli_1.0_test.jsonl',
        '../../data/snli_1.0/snli_1.0_test.jsonl',
    ]

    true_labels = None
    for path in test_file_paths:
        try:
            true_labels = pd.read_json(path, lines=True)['gold_label']
            print(f"Found test data at: {path}")
            break
        except FileNotFoundError:
            continue

    if true_labels is None:
        print("Error: Could not find test data file. Please ensure it exists in one of these locations:", *test_file_paths, sep='\n')
        return None

    # Calculate accuracy
    correct = (results == true_labels).sum()
    total = len(true_labels)
    accuracy = correct / total

    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy

# Usage example:
if __name__ == "__main__":
    accuracy = calculate_accuracy_from_json('longest_results_snli.json')
