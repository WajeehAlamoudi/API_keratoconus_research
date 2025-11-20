import json
import os


def build_prompt(image_path):
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
    return prompt


def load_or_create_json(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            results = json.load(f)
        return results
    except:
        print("⚠️ [WARNING] Output file corrupted. Reinitializing as empty array.")
        results = []
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results


def run_agent_ask(agent_class, agent_kwargs, output_file, image_folder):
    for image in os.listdir(image_folder):

        # check the image ends eith
        if not image.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"⚠️ [WARNING][{agent_class.__name__}] Passing {image} (not ends with .jpg .jpeg .png")
            continue

        # skip the already processed images
        results = load_or_create_json(output_file)
        processed_images = {entry["image_filename"] for entry in results if "image_filename" in entry}
        print(f"🔄 [INFO][{agent_class.__name__}] Already processed: {len(processed_images)} images")
        if image in processed_images:
            print(f"⏩ [INFO][{agent_class.__name__}] Skipping {image} (already processed)")
            continue

        # ready the agent prompt
        image_path = os.path.join(image_folder, image)
        prompt = build_prompt(image_path)

        # Create new agent (new chat session)
        try:
            agent = agent_class(**agent_kwargs)
        except Exception as e:
            print(f"‼️ [ERROR][{agent_class.__name__}] [{output_file}] INIT FAILED: {e}")
            return

        # run ask() isolated
        try:
            print(f"🔍 [INFO][{agent_class.__name__}] Processing {image}")
            response = agent.ask(prompt, response_type="json_object")
        except Exception as e:
            print(f"‼️ [ERROR][{agent_class.__name__}] but continue [{output_file}] ERROR on {image}: {e}")
            continue

        # parse json
        try:
            parsed = json.loads(response)
        except:
            try:
                cleaned = response.replace("```json", "").replace("```", "")
                parsed = json.loads(cleaned)
            except Exception as e:
                print(f"‼️ [ERROR][{agent_class.__name__}] but continue [{output_file}] JSON error {response}: {e}")
                continue

        # append result
        parsed = {"image_filename": image, **parsed}
        results.append(parsed)

        # save
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 [INFO][{agent_class.__name__}] [{output_file}] Saved: {image}")

    print(f"✅ [INFO][{agent_class.__name__}] [{output_file}] FINISHED ALL IMAGES")
