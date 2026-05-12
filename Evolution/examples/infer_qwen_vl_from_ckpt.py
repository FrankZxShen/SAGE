import argparse
import os
import subprocess
from typing import List

import torch
from PIL import Image
# from transformers import AutoModelForImageTextToText, AutoProcessor
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import json
import re



def ensure_hf_weights(actor_dir: str) -> str:
    actor_dir = os.path.abspath(actor_dir)
    hf_dir = os.path.join(actor_dir, "huggingface")
    os.makedirs(hf_dir, exist_ok=True)

    has_weights = any(
        os.path.exists(os.path.join(hf_dir, fname))
        for fname in ("pytorch_model.bin", "model.safetensors")
    )
    if has_weights:
        return hf_dir

    shard_exists = any(
        fname.startswith("model_world_size_") and fname.endswith("_rank_0.pt")
        for fname in os.listdir(actor_dir)
    )
    if not shard_exists:
        raise RuntimeError(
            f"未找到模型分片文件，请确认 --ckpt_dir 指向 'global_step_*/actor' 目录。当前: {actor_dir}"
        )

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "model_merger.py"))
    subprocess.run(["python3", script_path, "--local_dir", actor_dir], check=True)
    return hf_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default='/path/to//global_step_290_eta1.0', type=str)
    parser.add_argument("--prompt", default="Find the toilet.", type=str, help="The question or instruction.")
    parser.add_argument("--max_new_tokens", default=4096, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--data_dir", default="/path/to/CG-DATA-3/eps_00008-VYnUX657cVo/seed_8657", type=str, help="Path to the data directory containing JSON and images.")
    parser.add_argument("--images", nargs='+', default=['/path/to/geometry_data/image-1d100e9.png'], help="List of image paths")
    parser.add_argument("--labels", nargs='+', default=None, help="List of labels for each image")
    parser.add_argument("--use_exp", action="store_true", default=False, help="Whether to include experience in the prompt.")
    args = parser.parse_args()

    # Load data from directory if provided
    # if args.data_dir:
    #     data_dir = os.path.abspath(args.data_dir)
    #     seed_name = os.path.basename(data_dir)
    #     json_path = os.path.join(data_dir, f"{seed_name}.json")
        
    #     if not os.path.exists(json_path):
    #         # Try finding any json file
    #         json_files = glob.glob(os.path.join(data_dir, "*.json"))
    #         if json_files:
    #             json_path = json_files[0]
    #         else:
    #             raise FileNotFoundError(f"No JSON file found in {data_dir}")
        
    #     print(f"Loading data from {json_path}")
    #     with open(json_path, 'r') as f:
    #         data = json.load(f)
        
    #     # Extract question from the first point's eqa_experience
    #     # format: "IF answering \"Question?\" AND ..."
    #     question = None
    #     experience = None
    #     points = data.get("points", [])
    #     for point in points:
    #         exp = point.get("eqa_experience", "")
    #         if exp:
    #             experience = exp
    #             match = re.search(r'IF answering "(.*?)" AND', exp)
    #             if match:
    #                 question = match.group(1)
    #                 break
        
    #     if question:
    #         args.prompt = question
    #         print(f"Extracted question: {args.prompt}")
    #     else:
    #         print("Warning: Could not extract question from eqa_experience. Using default/provided prompt.")

    #     # Construct prompt intro
    #     exp_section = ""
    #     if experience and args.use_exp:
    #          exp_section = (
    #              "<EXP>\n"
    #              "Guidance from Memory:\n"
    #              f"{experience}\n"
    #              "(Instruction: Carefully check if any candidate image contains the visual cues mentioned in the 'IF' condition of this experience. If a match is found, strictly prioritize that path as per the 'THEN' rule.)\n"
    #              "</EXP>\n\n"
    #          )

    #     prompt_intro = (
    #         "Task: You are an intelligent robot navigating in an indoor scene. Your task is to select an image for further exploration to answer the given Question.\n\n"
    #         "Context:\n"
    #         "- You have access to images and their semantic parsings (objects).\n"
    #         "- You have access to a \"Navigation Experience\" derived from past successful behaviors.\n\n"
    #         f"Question:\n{args.prompt}\n\n"
    #         f"{exp_section}"
    #         "Definitions:\nObserving unexplored areas may yield new clues to answer the Question. Selecting an image means that you will further explore that direction. If you choose an image, you need to explain why you would like to choose that direction to explore. Below are the candidate images you can choose from.\n"
    #         "Candidate Images:\n"
    #     )
    #     image_paths = []
    #     labels_list = []
        
    #     for point in points:
    #         point_idx = point.get("point_idx", 0)
    #         point_dir = os.path.join(data_dir, f"point_{point_idx}")
            
    #         for view in point.get("views", []):
    #             filename = view.get("filename")
    #             if filename:
    #                 img_path = os.path.join(point_dir, filename)
    #                 # Check if file exists, sometimes filenames in json might not match exactly or be relative
    #                 if not os.path.exists(img_path):
    #                     # Try flat structure if point_dir doesn't exist
    #                     flat_path = os.path.join(data_dir, filename)
    #                     if os.path.exists(flat_path):
    #                         img_path = flat_path
                    
    #                 if os.path.exists(img_path):
    #                     image_paths.append(img_path)
    #                     # Construct labels string
    #                     detected_objs = view.get("all_2d_labels", [])
    #                     labels_str = ", ".join(detected_objs) if detected_objs else "None"
    #                     labels_list.append(labels_str)
    #                 else:
    #                     print(f"Warning: Image {filename} not found at {img_path}")

    #     if image_paths:
    #         # Ensure we have at least 4 images, repeat if necessary or slice if too many (though usually we want exactly what's there)
    #         # The user requested exactly 4 images.
    #         if len(image_paths) < 4:
    #             print(f"Warning: Found only {len(image_paths)} images, padding with the last image to reach 4.")
    #             while len(image_paths) < 4:
    #                 image_paths.append(image_paths[-1])
    #                 labels_list.append(labels_list[-1])
    #         elif len(image_paths) > 4:
    #             print(f"Warning: Found {len(image_paths)} images, taking the first 4.")
    #             image_paths = image_paths[:4]
    #             labels_list = labels_list[:4]

    #         args.images = image_paths
    #         args.labels = labels_list
    #         print(f"Loaded {len(image_paths)} images:")
    #         for idx, img_p in enumerate(image_paths):
    #             print(f"  Image {idx}: {img_p}")
    #     else:
    #         raise RuntimeError("No images found in the data directory.")

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    actor_dir = ckpt_dir if os.path.basename(ckpt_dir) == "actor" else os.path.join(ckpt_dir, "actor")
    if not os.path.isdir(actor_dir):
        alt = ckpt_dir  # 用户可能已直接传入 actor 目录
        if os.path.isdir(alt) and os.path.basename(alt) == "actor":
            actor_dir = alt
        else:
            raise RuntimeError(
                f"检查点目录不存在或结构不正确: {actor_dir}\n"
                "请传入形如 '/.../global_step_XX/actor' 或 '/.../global_step_XX'"
            )
    hf_dir = ensure_hf_weights(actor_dir)
    # hf_dir = actor_dir
    print(hf_dir)
    exit()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(hf_dir, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    
    image_paths = args.images
    content = []
    processed_images = []
    
    if args.data_dir and prompt_intro:
        content.append({"type": "text", "text": prompt_intro})
    
    labels = args.labels if args.labels else ["None"] * len(image_paths)
    if len(labels) < len(image_paths):
        labels = labels + ["None"] * (len(image_paths) - len(labels))

    for i, img_path in enumerate(image_paths):
        content.append({"type": "text", "text": f"\nImage {i}: "})
        content.append({"type": "image", "image": img_path})
        
        content.append({"type": "text", "text": f"\n[Detected Objects]: {labels[i]}\n"})
        
        # Load and resize image as in nav_dataset.py
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256), Image.BICUBIC)
        processed_images.append(image)

    prompt_instruction = (
        "\nInstruction for Reasoning:\n"
        "1. Analyze the <EXP>: Does any image match the visual condition described in the experience?\n"
        f"2. Analyze the Question: What visual evidence is needed to answer \"{args.prompt}\"?\n"
        "3. Crucial Constraint: The Reasoning must be a single, concise sentence, focusing only on the primary evidence (sighting or the applied Experience rule).\n"
        "4. Provide your image selection in following format: \"Reasoning: [Reason]\nImage i\", where i is the index of the image you choose.\n"
        "Now return your response in the following strict format:\n"
        "Reasoning: [Briefly explain your logic (MAX ONE SENTENCE). Example: \"Image 2 is chosen because it directly shows the toilet.\" or \"Image 1 is selected as it matches the EXP rule for finding a bed.\"]\n"
        "Selection: Image i\n"
    )
    content.append({"type": "text", "text": prompt_instruction})

    messages = [
        {"role": "user", "content": content}
    ]

    # Apply template (tokenize=False to get the prompt string)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Process inputs
    inputs = processor(images=processed_images, text=[prompt], add_special_tokens=False, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
