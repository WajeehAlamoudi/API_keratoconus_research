import os
import json
import config
from agents import OPENAIAGENT


image_folder_path = r"C:\Users\wajee\PycharmProjects\API_keratoconus_research\images"
output_file = "gpt5_results.json"


if not os.path.exists(output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")
first_entry = True

for image in os.listdir(image_folder_path):
    if not image.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_folder_path, image)
    print(f"\n🔍 Processing {image}")

    GPT_5_agent = OPENAIAGENT(openai_api_key=config.OPENAI_API_KEY, model="gpt-5")

    prompt = {
        "prompt": """
        Based on this photo, determine if keratoconus is present, then stage according to Amsler–Krumeich and Belin ABCD staging systems.
        Respond in JSON format following this structure:
        {
        "keratoconus_diagnosis": "",
        "justification": "",
        "amsler_krumeich_stage": "",
        "amsler_krumeich_basis": "",
        "belin_abcd_overall_stage": "",
        "belin_abcd_basis": ""
        }
        """,
        "images": [image_path],
    }

    try:
        response = GPT_5_agent.ask(user_input=prompt, response_type="json_object")
        print(f"✅ Raw model response: {response[:200]}...")

        # Parse the JSON safely
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            cleaned = response.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(cleaned)

        parsed = {"image_filename": image, **parsed}

        with open(output_file, "a", encoding="utf-8") as f:
            if first_entry:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
                first_entry = False
            else:
                f.write(",\n")
                json.dump(parsed, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved results for {image} to {output_file}")

    except Exception as e:
        print(f"❌ Error processing {image}: {e}")

# ✅ Close the JSON array properly at the end
with open(output_file, "a", encoding="utf-8") as f:
    f.write("\n]")

print(f"\n✅ All results written incrementally and saved to {output_file}")