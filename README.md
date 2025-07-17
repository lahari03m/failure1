# Technician Work Order Summarizer

Summarizes technician work orders in batches, predicts asset failures (30-60 days window), and generates master summaries using local LLMs via Ollama.

## Requirements
- Python 3.x
- Ollama installed on your machine
- Python dependencies:
```
pip install pandas
```

## Running the Summarizer
```
python3 work_order_summarizer.py --csv_file your_work_orders.csv --chunk_size 10 --temperature 0.5 --model llama3
```

### Arguments
| Option          | Description                       | Default |
|-----------------|-----------------------------------|---------|
| --csv_file      | Path to your input CSV file       | Required|
| --chunk_size    | Number of records per batch       | 10      |
| --temperature   | Creativity level for model (0-1)  | 0.5     |
| --model         | Ollama model to use (llama3, etc.)| llama3  |

## Outputs
- `batch_summaries/` - Individual JSON summaries per batch
- `all_batch_summaries.json` - Combined summaries
- `master_summary.json` - Master summarization across all batches
- `asset_failure_summary.json` - Per-asset failure statistics