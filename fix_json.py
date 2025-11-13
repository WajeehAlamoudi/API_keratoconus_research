import sys
import os
import json
import re

INVISIBLE_CHARS = [
    "\u200b",  # ZWSP
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\ufeff",  # BOM
    "\xa0",  # NBSP
]
SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"',  # “ ”
    "\u2018": "'", "\u2019": "'",  # ‘ ’
}


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def clean_invisibles(s: str) -> str:
    s = s.replace("ZWSPZWSP", "")
    for ch in INVISIBLE_CHARS:
        s = s.replace(ch, "")
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    return s


def extract_balanced_objects(raw: str):
    objs = []
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                escape = False
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        block = raw[start:i + 1]
                        objs.append((start, i, block))
                        start = None
    return objs


# ---------------------------------------------------------------------

def auto_fix_common_damage(txt: str) -> str:
    """
    Heuristics to repair common corruption:
    - flatten newlines/tabs
    - remove trailing commas before } or ]
    - fix stray ']' before '}'
    - insert missing commas between key-value pairs
    - convert single quotes to double quotes
    """
    flat = re.sub(r"[\r\n\t]+", " ", txt)
    flat = re.sub(r"\s{2,}", " ", flat)

    # Fix stray ']' before closing '}'
    flat = re.sub(r"\]\s*}", "}", flat)

    # Remove trailing commas before closing braces/brackets
    flat = re.sub(r",\s*([}\]])", r"\1", flat)

    # Insert missing commas between value tokens and next quoted key
    flat = re.sub(
        r'("(?:(?:\\.)|[^"\\])*"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null|\]|\})\s*(?="[^"]+"\s*:)',
        r'\1, ',
        flat
    )

    # Convert single-quoted strings → double-quoted strings
    flat = re.sub(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', flat)

    # Remove double commas like ", ," or ",, "
    flat = re.sub(r",\s*,", ",", flat)

    return flat


def try_parse_json(txt: str):
    try:
        return json.loads(txt), None
    except json.JSONDecodeError as e:
        return None, e


def save_invalid(idx: int, txt: str, err, folder: str, source_offset: int):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"invalid_{idx:04d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
        f.write("\n\n---\n")
        f.write(f"ERROR: {repr(err)}\n")
        f.write(f"SOURCE_OFFSET: {source_offset}\n")
    return path


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data_new_ai.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "fixed_data.json"
    invalid_dir = os.path.splitext(out_path)[0] + "_invalid"

    raw0 = read_text(in_path)
    raw = clean_invisibles(raw0)

    blocks = extract_balanced_objects(raw)
    print(f"🧩 Found {len(blocks)} balanced {{...}} blocks in raw TXT")

    valid_objs = []
    invalid_count = 0

    for idx, (start, end, block) in enumerate(blocks, 1):
        cleaned = clean_invisibles(block)

        # First try strict
        obj, err = try_parse_json(cleaned)
        if obj is not None:
            valid_objs.append(obj)
            continue

        # Then try heuristic repair
        repaired = auto_fix_common_damage(cleaned)
        obj2, err2 = try_parse_json(repaired)
        if obj2 is not None:
            print(f"🛠️  Auto-repaired object #{idx} at offset {start}")
            valid_objs.append(obj2)
            continue

        # Still invalid → save for manual review
        invalid_count += 1
        p = save_invalid(invalid_count, block, err2 or err, invalid_dir, start)
        print(f"⚠️ Could not repair object #{idx} (offset {start}) → saved {p}")

    # Write valid objects as array
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(valid_objs, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    print(f"✅ Valid objects written: {len(valid_objs)} → {out_path}")
    print(f"🗂️  Invalid objects saved: {invalid_count} → {invalid_dir}/invalid_XXXX.txt")


if __name__ == "__main__":
    main()
