import sqlite3
import pandas as pd

DB_PATH = "data/responses.db"

def load_labels(conn):
    return pd.read_sql_query("SELECT * FROM labels", conn)

def win_rate_by_temperature(conn):
    """Check if low-temp (0.3, focused) or high-temp (0.9, creative) responses win more"""
    responses = pd.read_sql_query("SELECT id, prompt_id, response, temperature FROM responses", conn)
    labels = pd.read_sql_query("SELECT prompt_id, chosen_response FROM labels", conn)

    results = []
    for _, label in labels.iterrows():
        if label["chosen_response"] == "TIE":
            continue
        match = responses[
            (responses["prompt_id"] == label["prompt_id"]) &
            (responses["response"] == label["chosen_response"])
        ]
        if not match.empty:
            results.append(match.iloc[0]["temperature"])

    temp_series = pd.Series(results)
    return temp_series.value_counts()

def export_preference_pairs(conn, output_path="data/preference_pairs.csv"):
    """Export in standard RLHF chosen/rejected format"""
    labels = pd.read_sql_query("SELECT prompt, chosen_response, rejected_response, rationale FROM labels WHERE chosen_response != 'TIE'", conn)
    labels.to_csv(output_path, index=False)
    print(f"Exported {len(labels)} preference pairs to {output_path}")

def main():
    conn = sqlite3.connect(DB_PATH)

    labels = load_labels(conn)
    print(f"\nTotal labels collected: {len(labels)}")

    ties = (labels["chosen_response"] == "TIE").sum()
    print(f"Ties: {ties}")
    print(f"Decisive judgments: {len(labels) - ties}")

    print("\n--- Win rate by temperature setting ---")
    print(win_rate_by_temperature(conn))
    print("(temp=0.3 = focused/consistent, temp=0.9 = creative/varied)")

    print("\n--- Sample rationales ---")
    for _, row in labels.head(5).iterrows():
        print(f"- {row['rationale']}")

    export_preference_pairs(conn)
    conn.close()

if __name__ == "__main__":
    main()
