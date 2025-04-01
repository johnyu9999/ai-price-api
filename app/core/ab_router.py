# app/core/ab_router.py

import hashlib

def choose_model_version(user_id: str, available_versions: list[str]) -> str:
    if not available_versions:
        raise ValueError("No available model versions to choose from.")

    hashed = hashlib.sha256(user_id.encode()).hexdigest()
    index = int(hashed, 16) % len(available_versions)
    return available_versions[index]
