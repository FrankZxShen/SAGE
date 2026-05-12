import os
import json
import random
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = None

    class Dataset:
        pass
from PIL import Image
from collections import defaultdict
import hashlib

class NavDataset(Dataset):
    def __init__(self, data_path, tokenizer, processor, max_prompt_length=1024, split='train', val_ratio=0.2, rollout_ratio=0.5, usage_ratio=1.0, use_three_images: bool = False):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.split = split
        self.val_ratio = val_ratio
        self.rollout_ratio = rollout_ratio
        self.usage_ratio = usage_ratio
        self.use_three_images = use_three_images
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # data_path is like /mnt/disk/shengzhixuan/project/NavEvolver_v2/DATA/results/CG-DATA-2
        # structure: eps_XXX/seed_XXX/seed_XXX.json and point_Y/point_Y.png, etc.
        
        # Walk through the directory structure
        if not os.path.exists(self.data_path):
            print(f"Warning: Data path {self.data_path} does not exist.")
            return []

        # Get all episode directories and sort them for deterministic behavior
        eps_dirs = sorted([d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))])
        
        for eps_dir in eps_dirs:
            eps_path = os.path.join(self.data_path, eps_dir)
            
            # Filter by usage_ratio (random 60% if usage_ratio=0.6)
            # Use a different salt for hash to be independent of val split if possible, 
            # but usually we want consistency.
            # Using "usage" as salt.
            usage_hash_val = int(hashlib.md5((eps_dir + "usage").encode()).hexdigest(), 16)
            usage_normalized = (usage_hash_val % 10000) / 10000.0
            
            if usage_normalized >= self.usage_ratio:
                continue

            # Hash eps_dir name to decide if it is train or val
            # Use MD5 or similar to be deterministic across runs
            hash_val = int(hashlib.md5(eps_dir.encode()).hexdigest(), 16)
            # Map hash to [0, 1]
            normalized_hash = (hash_val % 10000) / 10000.0
            
            is_val = normalized_hash < self.val_ratio
            
            if self.split == 'train' and is_val:
                continue
            if self.split == 'val' and not is_val:
                continue

            for seed_dir in os.listdir(eps_path):
                seed_path = os.path.join(eps_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue
                
                # Check for seed json file
                json_file = os.path.join(seed_path, f"{seed_dir}.json")
                if not os.path.exists(json_file):
                    continue
                
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error loading json {json_file}: {e}")
                    continue

                goals = []
                generated_question = data.get("generated_question", "")
                if generated_question:
                    goals.append(generated_question)
                
                eqa = data.get("eqa", "") or data.get("eqa_question", "")
                if eqa:
                    goals.append(eqa)

                # textgoal = data.get("textgoal", "")
                # if textgoal:
                #     goals.append(f"Where can I find {textgoal}?")

                objectgoal = data.get("objectgoal", "")
                if objectgoal:
                    goals.append(f"Where can I find {objectgoal}?")
                
                # Deduplicate and sort
                goals = sorted(list(set(goals)))

                points = data.get("points", [])
                
                if not goals or not points:
                    continue
                
                # Organize points by index for easy access
                points_map = {p["point_idx"]: p for p in points}
                sorted_indices = sorted(points_map.keys())
                target_idx = sorted_indices[-1]
                
                # Pre-process metadata map: {(point_idx, view_label): {"labels": ..., "scene_graph": ...}}
                # Views in JSON are: angle 0 -> center (point_X.png), -120 -> left/right?, 120 -> right/left?
                # Let's map filenames to metadata since filenames are explicit in JSON and we build paths from indices.
                # JSON structure: points -> views -> filename (e.g. "point_0.png")
                # We can map (point_idx, filename) -> metadata
                
                metadata_map = {}
                for p in points:
                    p_idx = p["point_idx"]
                    views = p.get("views", None)
                    if isinstance(views, list) and len(views) > 0:
                        for v in views:
                            fname = v.get("filename", "")
                            if not fname:
                                continue
                            labels = v.get("all_2d_labels", [])
                            if isinstance(labels, list):
                                labels_str = ", ".join([str(l) for l in labels])
                            else:
                                labels_str = str(labels)
                            sg_text = v.get("scene_graph_text", "")
                            metadata_map[(p_idx, fname)] = {"labels": labels_str, "scene_graph": sg_text}
                    else:
                        view_front = p.get("view_front", None)
                        if isinstance(view_front, dict):
                            fname = view_front.get("filename", "")
                            if fname:
                                labels = view_front.get("all_2d_labels", [])
                                if isinstance(labels, list):
                                    labels_str = ", ".join([str(l) for l in labels])
                                else:
                                    labels_str = str(labels)
                                sg_text = view_front.get("scene_graph_text", "")
                                metadata_map[(p_idx, fname)] = {"labels": labels_str, "scene_graph": sg_text}

                                if fname.endswith("_front.png"):
                                    left_name = fname[: -len("_front.png")] + "_left.png"
                                    right_name = fname[: -len("_front.png")] + "_right.png"
                                    left_path = os.path.join(seed_path, f"point_{p_idx}", left_name)
                                    right_path = os.path.join(seed_path, f"point_{p_idx}", right_name)
                                    if os.path.exists(left_path):
                                        metadata_map[(p_idx, left_name)] = {"labels": "", "scene_graph": ""}
                                    if os.path.exists(right_path):
                                        metadata_map[(p_idx, right_name)] = {"labels": "", "scene_graph": ""}

                # Create samples for each goal and point Y (except point 0, as we need Z < Y)
                for goal_text in goals:
                    for idx_y in sorted_indices:
                        # Need Z < Y, so Y must be > 0 (if indices start at 0)
                        # Actually, let's look at available indices less than Y
                        possible_z = [idx for idx in sorted_indices if idx < idx_y]
                        if not possible_z:
                            continue
                            
                        # Path to point Y images
                        # Structure seems to be: seed_path/point_Y/point_Y.png, point_Y_120.png, point_Y_-120.png
                        # Based on ls output: seed_path/point_0/point_0.png, etc.
                        
                        point_y_dir = os.path.join(seed_path, f"point_{idx_y}")
                        if not os.path.exists(point_y_dir):
                            continue
                            
                        img_y_name = f"point_{idx_y}.png"
                        img_y_120_name = f"point_{idx_y}_120.png"
                        img_y_neg120_name = f"point_{idx_y}_-120.png"

                        img_y_path = os.path.join(point_y_dir, img_y_name)
                        img_y_120_path = os.path.join(point_y_dir, img_y_120_name)
                        img_y_neg120_path = os.path.join(point_y_dir, img_y_neg120_name)

                        if not (os.path.exists(img_y_path) and os.path.exists(img_y_120_path) and os.path.exists(img_y_neg120_path)):
                            img_y_name = f"point_{idx_y}_front.png"
                            img_y_120_name = f"point_{idx_y}_left.png"
                            img_y_neg120_name = f"point_{idx_y}_right.png"
                            img_y_path = os.path.join(point_y_dir, img_y_name)
                            img_y_120_path = os.path.join(point_y_dir, img_y_120_name)
                            img_y_neg120_path = os.path.join(point_y_dir, img_y_neg120_name)
                        
                        if not (os.path.exists(img_y_path) and os.path.exists(img_y_120_path) and os.path.exists(img_y_neg120_path)):
                            continue

                        # We need one Z < Y
                        # For training we might want to be dynamic, but Dataset usually pre-computes or computes on fly.
                        # Here we store enough info to sample Z on the fly or pre-generate pairs.
                        # The prompt requires 4 images: Y, Y_120, Y_-120, Z.
                        # Y is the answer (img0 in the description logic, but prompt order is shuffled).
                        # Wait, description says: 
                        # "image 0:<img0>, image 1:<img1>, image 2:<img2>, image 3:<img3>"
                        # "select a scene image that can answer this question"
                        # "Answer: image 0" (if image 0 is the correct one).
                        # The user says: "point_Y.png作为anwser(<img0>)"
                        # "point_Y_120.png（<img1>）和point_Y_-120.png（<img2>）以及eps中随机的point_Z（<img3>）（Z<Y）"
                        # "image的id会被打乱" -> This means the mapping from "image X" to the actual image content should be shuffled.
                        
                        # Retrieve metadata for Y images
                        meta_y = metadata_map.get((idx_y, img_y_name), {"labels": "", "scene_graph": ""})
                        meta_y_120 = metadata_map.get((idx_y, img_y_120_name), {"labels": "", "scene_graph": ""})
                        meta_y_neg120 = metadata_map.get((idx_y, img_y_neg120_name), {"labels": "", "scene_graph": ""})

                        retrieved_experience = data.get("retrieved_experience", "None")
                        if retrieved_experience in (None, "", "None"):
                            point_info = points_map.get(idx_y, {})
                            if goal_text == eqa:
                                retrieved_experience = point_info.get("eqa_experience", "None")
                            else:
                                retrieved_experience = point_info.get("objectgoal_experience", "None")

                        if goal_text == eqa:
                            answer = data.get("answer", "") or data.get("eqa_answer", "")
                        else:
                            answer = data.get("answer", "")

                        sample = {
                            "eps_path": eps_path,
                            "seed_path": seed_path,
                            "generated_question": goal_text,
                            "answer": answer,
                            "retrieved_experience": retrieved_experience,
                            "y_idx": idx_y,
                            "target_idx": target_idx,
                            "y_images": {
                                "center": {"path": img_y_path, "meta": meta_y, "name": img_y_name},
                                "left": {"path": img_y_120_path, "meta": meta_y_120, "name": img_y_120_name},  # Assuming 120 is left/right
                                "right": {"path": img_y_neg120_path, "meta": meta_y_neg120, "name": img_y_neg120_name}
                            },
                            "possible_z_indices": possible_z,
                            "metadata_map": metadata_map # Pass full map to retrieve Z metadata later
                        }

                        samples.append(sample)
                        
                        # Upsample Memory samples to balance with Frontier samples
                        if idx_y == target_idx:
                            # There are len(sorted_indices) points total.
                            # Frontier samples come from indices [1, target_idx - 1] -> Count = len - 2 (since 0 is skipped)
                            # Memory samples come from index target_idx -> Count = 1
                            # We want Total Memory = Total Frontier = len - 2
                            # So we need to add (len - 2) - 1 = len - 3 extra copies.
                            
                            num_extra = max(0, len(sorted_indices) - 3)
                            for _ in range(num_extra):
                                samples.append(sample.copy())
                        # print(sample)
                        # exit()
                    
        print(f"Loaded {len(samples)} samples from {self.data_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if torch is None:
            raise ModuleNotFoundError("torch is required to use NavDataset.__getitem__")
        
        # Select random Z
        z_idx = random.choice(sample["possible_z_indices"])
        
        # Find Z image (random view? user says "point_Z")
        # Usually point_Z refers to point_Z.png (center view) if not specified.
        # Structure: seed_path/point_Z/point_Z.png
        z_dir = os.path.join(sample["seed_path"], f"point_{z_idx}")
        img_z_name = f"point_{z_idx}.png"
        img_z_path = os.path.join(z_dir, img_z_name)
        if not os.path.exists(img_z_path):
            img_z_name = f"point_{z_idx}_front.png"
            img_z_path = os.path.join(z_dir, img_z_name)
        if not os.path.exists(img_z_path):
            try:
                candidates = [f for f in os.listdir(z_dir) if f.endswith(".png")]
            except Exception:
                candidates = []
            if candidates:
                img_z_name = candidates[0]
                img_z_path = os.path.join(z_dir, img_z_name)
        
        # Get Z metadata
        meta_z = sample["metadata_map"].get((z_idx, img_z_name), {"labels": "", "scene_graph": ""})
        
        # Prepare the 4 images
        # img0: Y (Answer)
        # img1: Y_120
        # img2: Y_-120
        # img3: Z
        
        # Note: The user prompt template says:
        # image 0:<img0>, 
        # image 1:<img1>, 
        # image 2:<img2>, 
        # image 3:<img3>, 
        # Answer: image 0
        # BUT also says "image的id会被打乱".
        # So we have 4 candidates. One is correct (Y).
        
        all_images = []
        for (p_idx, fname), meta in sample["metadata_map"].items():
            point_dir = os.path.join(sample["seed_path"], f"point_{p_idx}")
            path = os.path.join(point_dir, fname)
            if not os.path.exists(path):
                continue
            if fname.endswith("_120.png"):
                view_label = "120"
            elif fname.endswith("_left.png"):
                view_label = "120"
            elif fname.endswith("_-120.png"):
                view_label = "-120"
            elif fname.endswith("_right.png"):
                view_label = "-120"
            else:
                view_label = "center"
            all_images.append({"path": path, "meta": meta, "label": view_label, "point_idx": p_idx})

        gt = {
            "path": sample["y_images"]["center"]["path"],
            "meta": sample["y_images"]["center"]["meta"],
            "label": "center",
            "point_idx": sample["y_idx"],
            "is_answer": True,
        }

        pool = [img for img in all_images if not (img["path"] == gt["path"])]
        if len(pool) >= 3:
            sampled = random.sample(pool, 3)
        else:
            sampled = pool.copy()
            fallback = [
                {"path": sample["y_images"]["left"]["path"], "meta": sample["y_images"]["left"]["meta"], "label": "120", "point_idx": sample["y_idx"]},
                {"path": sample["y_images"]["right"]["path"], "meta": sample["y_images"]["right"]["meta"], "label": "-120", "point_idx": sample["y_idx"]},
                {"path": img_z_path, "meta": meta_z, "label": "center", "point_idx": z_idx},
            ]
            for f in fallback:
                if len(sampled) >= 3:
                    break
                if all(x["path"] != f["path"] for x in sampled) and f["path"] != gt["path"]:
                    sampled.append(f)

        def classify(view_label, p_idx, y_idx):
            if view_label in ("120", "-120") and p_idx == y_idx:
                return "Frontier Image"
            if view_label == "center" and p_idx < y_idx:
                return "Frontier Image"
            if view_label == "center" and p_idx == y_idx:
                # Only Memory if it is the target
                if y_idx == sample.get("target_idx", -1):
                    return "Memory Image"
                else:
                    return "Frontier Image"
            return "Memory Image"

        candidates = []
        gt["category"] = classify(gt["label"], gt["point_idx"], sample["y_idx"])
        candidates.append(gt)
        for img in sampled:
            candidates.append({
                "path": img["path"],
                "meta": img["meta"],
                "is_answer": False,
                "label": img["label"],
                "point_idx": img["point_idx"],
                "category": classify(img["label"], img["point_idx"], sample["y_idx"]),
            })

        if self.use_three_images:
            non_answer_indices = [i for i, c in enumerate(candidates) if not c["is_answer"]]
            if len(non_answer_indices) > 0:
                drop_idx = random.choice(non_answer_indices)
                candidates.pop(drop_idx)
        
        # Shuffle
        random.shuffle(candidates)
        
        # Split into categories for prompt display
        frontier_imgs = []
        memory_imgs = []
        for c in candidates:
            if c["category"] == "Frontier Image":
                frontier_imgs.append(c)
            else:
                memory_imgs.append(c)
        
        # Reconstruct display order: Frontier then Memory
        # The indices i in "Frontier Image i" and "Memory Image i" are local to the list.
        # But we need to track where the answer is.
        
        ground_truth = ""
        
        # Construct Prompt
        # "Problem:\nGiven the following scene image:\nimage 0:<img0>,\nimage 1:<img1>,\nimage 2:<img2>,\nimage 3:<img3>,\nyour task is to select a scene image that can answer this question:\n<generated_questions>\nThe image you choose must either answer this question or be closer to answering it than the given image.\n \nAnswer:\n"
        
        # In Qwen-VL, <image> tokens are used. 
        # We will construct a message list for the processor.
        
        # Modified Prompt
        retrieved_experience = sample.get('retrieved_experience', 'None')

        # Logic to determine add_exp based on rollout_ratio
        should_add_exp = random.random() < self.rollout_ratio
        
        if should_add_exp and retrieved_experience != 'None':
             exp_section = (
                 "<EXP>\n"
                 "Guidance from Memory:\n"
                 f"{retrieved_experience}\n"
                 "(Instruction: Carefully check if any candidate image contains the visual cues mentioned in the 'IF' condition of this experience. If a match is found, strictly prioritize that path as per the 'THEN' rule.)\n"
                 "</EXP>\n\n"
             )
        else:
             exp_section = ""

        prompt_intro = (
            "Task: You are an intelligent robot navigating in an indoor scene. Your task is to select a Frontier Image for further exploration or a Memory Image for answering the given Question.\n\n"
            # "Context:\n"
            # "- You have access to images and their semantic parsings (objects).\n"
            # "- You have access to a \"Navigation Experience\" derived from past successful behaviors.\n\n"
            f"Question:\n{sample['generated_question']}\n\n"
            f"{exp_section}"
            "Definitions:\n"
            "1. Frontier Image: An observation of unexplored areas that may provide new clues for answering the Question. Selecting a Frontier Image means that you will further explore that direction. If you choose a Frontier image, you need to explain why you would like to choose that direction to explore.\n"
            "2. Memory Image: An observation of several known objects. Selecting a Memory Image means that you have found the final destination to answer the Question. If you choose a Memory Image, you need to directly give an answer to the question. If you don't have enough information to give an answer, then don't choose a Memory Image.\n"
            "Candidate Images:\n"
        )
        content = [{"type": "text", "text": prompt_intro}]

        images_list = []
        
        # List Frontier Images
        if frontier_imgs:
            content.append({"type": "text", "text": "\n[Frontier Images]\n"})
            for i, img in enumerate(frontier_imgs):
                content.append({"type": "text", "text": f"Frontier Image {i}: "})
                content.append({"type": "image", "image": img["path"]})
                labels_text = img["meta"].get("labels", "None")
                content.append({"type": "text", "text": f"\n[Detected Objects]: {labels_text}\n"})
                images_list.append(img["path"])
                
                if img["is_answer"]:
                    ground_truth = f"Frontier Image {i}\nReason: {sample.get('answer', 'Unknown')}"
        else:
            content.append({"type": "text", "text": "\n[Frontier Images]\n"})
            content.append({"type": "text", "text": "\nNo Frontier Images available.\n"})

        # List Memory Images
        if memory_imgs:
            content.append({"type": "text", "text": "\n[Memory Images]\n"})
            for i, img in enumerate(memory_imgs):
                content.append({"type": "text", "text": f"Memory Image {i}: "})
                content.append({"type": "image", "image": img["path"]})
                labels_text = img["meta"].get("labels", "None")
                content.append({"type": "text", "text": f"\n[Detected Objects]: {labels_text}\n"})
                images_list.append(img["path"])
                
                if img["is_answer"]:
                    ground_truth = f"Memory Image {i}\nAnswer: {sample.get('answer', '')}"
        else:
            content.append({"type": "text", "text": "\n[Memory Images]\n"})
            content.append({"type": "text", "text": "\nNo Memory Images available.\n"})
            
        prompt_instruction = (
            "\nInstruction for Reasoning or Answer:\n"
            f"1. Find visual evidence for \"{sample['generated_question']}\" considering <EXP>.\n"
            "2. Response constraints:\n"
            "   - The Reason or Answer must be a simple, direct, natural sentence understandable by others. NO meta-words (e.g., 'memory', 'this image').\n"
            "   - You should always choose the ONE most relevant Memory/Frontier index, even if you used multiple images to infer the information.\n"
            "3. Provide your response in the following strict format:\n"
            "   - If you choose a Frontier Image (for reasoning/exploration), return: \"Frontier Image i\nReason: [Reason]\"\n"
            "   - If you choose a Memory Image (for answering), return: \"Memory Image i\nAnswer: [Answer]\"\n"
            "   (Where i is the index of the chosen image within its category)\n"
            "4. Examples:\n"
            "Frontier Image 0\nReason: The hallway likely leads to the living room.\n"
            "Memory Image 1\nAnswer: The red apple is on the white counter.\n"
            "Now return your response\n"
        )
        content.append({"type": "text", "text": prompt_instruction})
        
        # Prepare conversation
        messages = [
            {"role": "user", "content": content}
        ]
        
        # Apply template
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # Load images
        processed_images = []
        for img_path in images_list:
            image = Image.open(img_path).convert("RGB")
            # Resize to 224x224
            image = image.resize((256, 256), Image.BICUBIC)
            processed_images.append(image)
            # Return a dummy or fail?
            # Create a black image
            # processed_images.append(Image.new('RGB', (256, 256), color='black'))

        # Process inputs
        # Qwen2-VL processor expects 'images' (list of PIL) and 'text' (list of strings)
        # or conversation format.
        # If we use `image_file` in content, apply_chat_template might not handle it if we passed tokenize=False.
        # The prompt string will have <image> tokens.
        
        # Standard verl pipeline:
        # 1. apply_chat_template -> prompt string with <image> placeholders (or whatever the model uses)
        # 2. processor(images=images, text=[prompt], ...)
        
        model_inputs = self.processor(images=processed_images, text=[prompt], add_special_tokens=False, return_tensors="pt")
        
        input_ids = model_inputs['input_ids'][0]
        attention_mask = model_inputs['attention_mask'][0]
        pixel_values = model_inputs.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values
        
        image_grid_thw = model_inputs.get('image_grid_thw', None)

        # Handle position_ids if Qwen2-VL/Qwen3-VL
        # verl dataset does complex stuff for mrope.
        # We should probably try to reuse that logic or simplify it.
        # Ideally, we shouldn't implement this Dataset from scratch if we can reuse RLHFDataset, 
        # but the data loading logic (Y, Y_120, Z, random selection) is too specific.
        # So we must implement __getitem__ carefully.
        
        # Let's see how verl calculates position_ids.
        # It imports get_rope_index.
        
        position_ids = None
        if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
             if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
             else:
                from verl.models.transformers.qwen2_vl import get_rope_index
            
             vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
             text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
             position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
             position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        # Post-process (pad/truncate)
        # verl uses VF.postprocess_data
        from verl.utils import torch_functional as VF
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right" # or error
        )
        
        # Check truncation again after postprocess
        if len(input_ids) > self.max_prompt_length:
             print(f"Warning: input_ids length {len(input_ids)} > max_prompt_length {self.max_prompt_length}. This should not happen after postprocess if max_length is respected.")
        
        # Crucial check for image tokens vs features mismatch
        if pixel_values is not None:
            # Qwen2-VL uses specific image token id
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>") # Check if this is correct for Qwen3?
            # For Qwen2.5-VL / Qwen3-VL it might be different. 
            # Let's get it from config or processor if possible.
            if hasattr(self.processor, "image_token_id"):
                 image_token_id = self.processor.image_token_id
            elif hasattr(self.processor.tokenizer, "image_token_id"):
                 image_token_id = self.processor.tokenizer.image_token_id
            
            # Count image tokens in FINAL input_ids
            num_image_tokens = (input_ids == image_token_id).sum().item()
            # Calculate expected features (assuming spatial merge)
            # This is hard because features depend on grid_thw
            # But we can check if we lost any image tokens compared to before truncation?
            
            # Better approach:
            # If we truncated input_ids, we might have cut off some image tokens.
            # But pixel_values contains ALL images.
            # So if input_ids has fewer image tokens than pixel_values expects, we crash.
            
            # How many images did we load?
            num_images_loaded = len(processed_images)
            
            # If input_ids was truncated, we might have lost the tokens for the last image(s).
            # We must sync pixel_values to match the remaining image tokens.
            # BUT Qwen-VL is complex: one image -> many tokens.
            # If we cut a few tokens from an image, the whole image is invalid?
            # Or does it just need the tokens to match features?
            # The features are calculated per image.
            # 1 image -> N features.
            # 1 image -> N tokens.
            # They must match exactly.
            
            # If we truncate in the middle of an image's tokens, we are broken.
            # If we truncate an entire image, we must remove its pixel_values.
            
            # Given the error is "tokens: 14520, features 30720", it seems we have FEWER tokens than features.
            # This means input_ids was truncated (or tokens lost), but pixel_values still has all features.
            
            # We MUST avoid truncation of image tokens.
            # OR we must filter samples that are too long.
            
            # Current max_prompt_length is 8192.
            # The error says 14520 tokens vs 30720 features.
            # Wait, 14520 tokens is > 8192.
            # Why did we get 14520 tokens if we truncated to 8192?
            # Ah, the error "tokens: 14520" might be from a previous run or I misread?
            # User said: "tokens: 14520, features 30720".
            # 14520 is definitely > 8192.
            # Maybe postprocess_data didn't truncate?
            # Or maybe the error comes from `_get_input_embeds` which uses `input_ids` PASSED to it.
            
            # If `input_ids` has 14520 tokens, then `max_prompt_length` was ignored or not effective?
            # VF.postprocess_data should truncate.
            
            # Let's look at features: 30720.
            # 30720 features / 4 images = 7680 features/image.
            # 7680 = 16 * (H/14 * W/14)? 
            # If 360x360: 14x14 patch -> 25x25 grid. 625 patches.
            # Qwen2-VL logic is complex.
            
            # But 30720 is huge.
            # 14520 is also huge.
            
            # If the error persists, it means we are sending mismatched data.
            # If we cannot easily truncate safely (because of partial image token removal),
            # we should probably just SKIP this sample if it's too long.
            
            if len(model_inputs['input_ids'][0]) > self.max_prompt_length:
                pass


        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # Truncate raw prompt if needed (though we just processed input_ids)
        if len(raw_prompt_ids) > self.max_prompt_length:
             raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]

        # Ground Truth
        
        exp_mask = 1.0 if should_add_exp else 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "ground_truth": ground_truth,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "multi_modal_data": {"images": processed_images},
            "eps_path": sample['eps_path'],
            "seed_path": sample['seed_path'],
            "y_idx": sample['y_idx'],
            "exp_mask": torch.tensor([exp_mask], dtype=torch.float32)
        }
