import json
from datetime import datetime
from app.core.model_holder import model_holder

def log_switch_to_file(from_version, to_version, duration_ms):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": "switch",
        "from": from_version,
        "to": to_version,
        "duration_ms": duration_ms
    }
    with open("logs/model_switch_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def log_predict_latency(duration_ms: float):
    model_holder.total_latency += duration_ms
    model_holder.total_requests += 1
