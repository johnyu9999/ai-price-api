from threading import Lock
from collections import defaultdict

class ModelHolder:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.input_dims = 3
        self.model_lock = Lock()

        self.hit_counter = defaultdict(int)
        self.error_count = 0
        self.total_latency = 0.0
        self.total_requests = 0

        self.last_switch_info = {}
        self.available_versions = {"v1", "v2", "v3", "v4", "v5"}
        self.frozen_versions = set()

model_holder = ModelHolder()
