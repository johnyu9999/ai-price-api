# model_holder.py

import threading
from datetime import datetime
from collections import deque
from collections import defaultdict


model = None
model_versions = {}
current_version = "v1"
model_lock = threading.Lock()

startup_time = datetime.utcnow()
last_switch_info = {
    "from": None,
    "to": None,
    "duration_ms": None,
    "timestamp": None
}

frozen_versions = set()

prediction_durations_ms = deque(maxlen=100)

predict_error_total = 0

predict_latency_total = defaultdict(float)
predict_latency_count = defaultdict(int)
model_hit_counter = {}
