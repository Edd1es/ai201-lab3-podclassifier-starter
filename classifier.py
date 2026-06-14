import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    instruction = (
        "You are classifying podcast episodes by their format. "
        "Classify the episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n\n"
        "Return your answer in exactly this format:\n"
        "Label: <one of: interview, solo, panel, narrative>\n"
        "Reasoning: <one or two sentences>\n"
    )

    if not labeled_examples:
        examples_block = "(no labeled examples provided)"
    else:
        examples_block = "\n\n---\n\n".join(
            f"Title: {ex['title']}\n"
            f"Description: {ex['description']}\n"
            f"Label: {ex['label']}"
            for ex in labeled_examples
        )

    new_episode = f"Title: (unknown)\nDescription: {description}\nLabel: ?"

    return (
        f"{instruction}\n"
        f"Here are labeled examples:\n\n{examples_block}\n\n---\n\n"
        f"Now classify this new episode:\n\n{new_episode}"
    )


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = response.choices[0].message.content

        # Parse label
        label = "unknown"
        reasoning = text.strip()
        for line in text.splitlines():
            cleaned = line.strip().lower().lstrip("*-` ").rstrip("*` .,:")
            if cleaned.startswith("label:"):
                candidate = cleaned.split("label:", 1)[1].strip().strip("*`. ,")
                if candidate in VALID_LABELS:
                    label = candidate
                break

        # Fallback: scan for any bare valid label
        if label == "unknown":
            lowered = text.lower()
            for valid in VALID_LABELS:
                if valid in lowered:
                    label = valid
                    break

        # Pull reasoning line if present
        for line in text.splitlines():
            if line.strip().lower().startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
                break

        return {"label": label, "reasoning": reasoning}
    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error: {e}"}
