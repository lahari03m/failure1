import pandas as pd
import json
import subprocess
import os
import argparse

def summarize_with_ollama(prompt, model, temperature):
    command = f"echo \"{prompt}\" | ollama run {model} --temperature {temperature}"
    result = subprocess.check_output(command, shell=True, text=True)
    return result.strip()

def main(csv_file, chunk_size, temperature, model):
    BATCH_SUMMARY_FOLDER = 'batch_summaries'
    os.makedirs(BATCH_SUMMARY_FOLDER, exist_ok=True)

    df = pd.read_csv(csv_file)

    required_columns = ['Work Order ID', 'Asset ID', 'Failure Description', 'Resolution']
    optional_columns = ['Technician Comments']

    for col in required_columns:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    columns_to_use = required_columns + [col for col in optional_columns if col in df.columns]
    num_chunks = len(df) // chunk_size
    all_batch_summaries = []

    for batch_num in range(num_chunks):
        start = batch_num * chunk_size
        end = start + chunk_size
        batch_df = df.iloc[start:end][columns_to_use]

        asset_failures = []
        for asset_id, group in batch_df.groupby('Asset ID'):
            failures = group['Failure Description'].dropna().unique().tolist()
            asset_failures.append({
                "Asset ID": asset_id,
                "Failures": failures
            })

        prompt = (
            f"Given these technician work orders with asset IDs, failure descriptions, resolutions, "
            f"and technician comments if present:\n\n{batch_df.to_string(index=False)}\n\n"
            f"For each asset:\n"
            f"- Summarize its failure patterns.\n"
            f"- Predict in how many days it might fail again (only if possible from data, between 30-60 days max).\n"
            f"- If data is insufficient to predict failure, explicitly state that.\n"
            f"- Provide actionable recommendations to avoid future failures.\n"
            f"Do not hallucinate or estimate beyond the data provided."
        )

        batch_summary_text = summarize_with_ollama(prompt, model, temperature)

        batch_summary = {
            "batch_number": batch_num + 1,
            "assets_failure_details": asset_failures,
            "summary": batch_summary_text
        }

        all_batch_summaries.append(batch_summary)

        batch_file = f'{BATCH_SUMMARY_FOLDER}/batch_summary_{batch_num + 1}.json'
        with open(batch_file, 'w') as f:
            json.dump(batch_summary, f, indent=4)

        print(f"✅ Saved {batch_file}")

    with open('all_batch_summaries.json', 'w') as f:
        json.dump({"batches": all_batch_summaries}, f, indent=4)
    print("✅ Combined batch summaries saved to all_batch_summaries.json")

    all_summaries_text = "\n\n".join([f"Batch {b['batch_number']} Summary: {b['summary']}" for b in all_batch_summaries])
    master_prompt = (
        f"Here are the summaries of technician work orders batches. "
        f"Generate a comprehensive master summary that captures trends, recurring asset failures, "
        f"and preventive maintenance suggestions based on the data provided only.\n\n"
        f"{all_summaries_text}"
    )

    master_summary_text = summarize_with_ollama(master_prompt, model, temperature)

    with open('master_summary.json', 'w') as f:
        json.dump({"master_summary": master_summary_text}, f, indent=4)

    print("✅ Master summary saved to master_summary.json")

    asset_summary = []
    for asset_id, group in df.groupby('Asset ID'):
        failures = group['Failure Description'].dropna().tolist()
        unique_failures = list(set(failures))
        asset_summary.append({
            "Asset ID": asset_id,
            "total_failures": len(failures),
            "common_failures": unique_failures
        })

    with open('asset_failure_summary.json', 'w') as f:
        json.dump({"asset_summary": asset_summary}, f, indent=4)

    print("✅ Per-asset failure summary saved to asset_failure_summary.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize technician work orders with predictions.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the work orders CSV file')
    parser.add_argument('--chunk_size', type=int, default=10, help='Number of records per batch')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for LLM (0 to 1)')
    parser.add_argument('--model', type=str, default='llama3', help='Ollama model name (e.g., llama3, mistral)')

    args = parser.parse_args()
    main(args.csv_file, args.chunk_size, args.temperature, args.model)