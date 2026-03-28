import pandas as pd
import json

def summarize_results(results_path: str = "results_db.jsonl") -> dict:
    data = []
    with open(results_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    if not data:
        return {}
        
    df = pd.DataFrame(data)
    
    summary_df = df.groupby(["method", "k", "c", "dataset", "metric_name"]).agg(
        mean_score=('metric_value', 'mean'),
        std_score=('metric_value', 'std'),
        avg_latency_ms=('latency_ms', 'mean'),
        peak_vram_mb=('mem_mb', 'max')
    ).reset_index()
    
    results_dict = summary_df.to_dict(orient="records")
    return {"aggregated": results_dict}

if __name__ == "__main__":
    print("Summary:")
    print(summarize_results())
