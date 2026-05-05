user_memory = {}

def save_preference(user_id: str, key: str, value: str):
    if user_id not in user_memory:
        user_memory[user_id] = {}
    user_memory[user_id][key] = value

def get_preference(user_id: str):
    return user_memory.get(user_id, {})