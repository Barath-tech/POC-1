import json
import os

def load_memory(memory_path):
    """Load memory (categories and their embeddings) from JSON file."""
    print(f"[load_memory] Loading memory from: {memory_path}")
    if not os.path.exists(memory_path):
        print("[load_memory] File not found. Creating new memory with categories:{}")
        return {"categories": {}}

    with open(memory_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            print("[load_memory] Memory loaded successfully.")
            return data
        except json.JSONDecodeError:
            print("[load_memory] Corrupt JSON. Reinitializing memory.")
            return {"categories": {}}


def save_memory( memory_path, memory):
    """Save memory back to JSON file."""
    print(f"[save_memory] Saving memory to: {memory_path}")
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    print("[save_memory] Memory saved.")


def add_category(memory, category_name, example, embedding):
    """
    category_name ➝ must come from GPT (string)
    example ➝ incident ID or text
    embedding ➝ vector list
    """
    category_name = str(category_name).strip()   # ✅ force string key

    if category_name not in memory["categories"]:
        memory["categories"][category_name] = {
            "examples": example,
            "embedding": embedding
        }
    else:
        # append example if category already exists
        existing_examples = memory["categories"][category_name]["examples"]
        if isinstance(existing_examples, str) and existing_examples:
            memory["categories"][category_name]["examples"] += f", {example}"
        else:
            memory["categories"][category_name]["examples"] = example

    return memory

def update_category(memory, category_name, example_id):
    """Update category with a new example ID (if not already present)."""
    if category_name in memory["categories"]:
        if example_id not in memory["categories"][category_name]["examples"]:
            memory["categories"][category_name]["examples"].append(example_id)
            print(f"[update_category] Added example {example_id} to category {category_name}")
    else:
        print(f"[update_category] Category {category_name} not found. Cannot update.")
