import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
import torch
import clip
import chromadb
from src.const import *




# For local VLM
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{c[1]}",
                        "detail": "high",
                    },
                }
            )
    return formated_content


# send information to openai
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,  # model = "deployment_name"
                messages=message_text,
                temperature=1.0,
                max_tokens=256,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 3s")
            time.sleep(3)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(3)
            retry_count += 1
            continue

    return None

# send information to qwen3vl
def call_openai_api_qwen3vlplus(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="qwen3-vl-plus-2025-09-23",  # model = "deployment_name"
                messages=message_text,
                temperature=1.0,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 3s")
            time.sleep(3)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(3)
            retry_count += 1
            continue

    return None  


# send information to openai-GPT4o
def call_openai_api_gpt4o(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",  # model = "deployment_name"
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 3s")
            time.sleep(3)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(3)
            retry_count += 1
            continue

    return None


# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal


def get_step_info(step, verbose=False):
    # 1 get question data
    question, image_goal = format_question(step)

    # 2 get step information(egocentric, frontier, snapshot)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))
    frontier_classes = step.get("frontier_classes", [])

    # 2.3 get snapshots
    snapshot_imgs, snapshot_classes = [], []
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]
        snapshot_imgs.append(encode_tensor2base64(snapshot_img))
        snapshot_class = [obj_map[int(sid)] for sid in step["snapshot_objects"][rgb_id]]
        # remove duplicates
        snapshot_class = sorted(list(set(snapshot_class)))
        seen_classes.update(snapshot_class)
        snapshot_classes.append(snapshot_class)

    # 3 prefiltering, note that we need the obj_id_mapping
    keep_index = list(range(len(snapshot_imgs)))
    if step.get("use_prefiltering") is True:
        n_prev_snapshot = len(snapshot_imgs)
        snapshot_classes, keep_index = prefiltering(
            question,
            snapshot_classes,
            seen_classes,
            step["top_k_categories"],
            image_goal,
            verbose,
        )
        snapshot_imgs = [snapshot_imgs[i] for i in keep_index]
        if verbose:
            logging.info(
                f"Prefiltering snapshot: {n_prev_snapshot} -> {len(snapshot_imgs)}"
            )

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        frontier_classes,
        snapshot_imgs,
        snapshot_classes,
        keep_index,
    )


def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. To answer the question, you are required to choose either a Snapshot as the answer or a Frontier to further explore.\n"
    sys_prompt += "Definitions:\n"
    sys_prompt += "Snapshot: A focused observation of several objects. Choosing a Snapshot means that this snapshot image contains enough information for you to answer the question. "
    sys_prompt += "If you choose a Snapshot, you need to directly give an answer to the question. If you don't have enough information to give an answer, then don't choose a Snapshot.\n"
    sys_prompt += "Frontier: An observation of an unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction. "
    sys_prompt += "If you choose a Frontier, you need to explain why you would like to choose that direction to explore.\n"

    content = []
    # 1 first is the question
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Frontier/Snapshot that would help find the answer of the question.\n"
    content.append((text,))

    # 2 add egocentric view
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))

    # 3 here is the snapshot images
    text = "The followings are all the snapshots that you can choose (followed with contained object classes)\n"
    text += "Please note that the contained classes may not be accurate (wrong classes/missing classes) due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make decisions.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i in range(len(snapshot_imgs)):
            content.append((f"Snapshot {i} ", snapshot_imgs[i]))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append(("\n",))

    # 4 here is the frontier images
    text = "The followings are all the Frontiers that you can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))

    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i\n[Answer]' or 'Frontier i\n[Reason]', where i is the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the first snapshot, you can return 'Snapshot 0\nThe fruit bowl is on the kitchen counter.'. "
    text += "If you choose the second frontier, you can return 'Frontier 1\nI see a door that may lead to the living room.'.\n"
    text += "Note that if you choose a snapshot to answer the question, (1) you should give a direct answer that can be understood by others. Don't mention words like 'snapshot', 'on the left of the image', etc; "
    text += "(2) you can also utilize other snapshots, frontiers and egocentric views to gather more information, but you should always choose one most relevant snapshot to answer the question.\n"
    content.append((text,))

    return sys_prompt, content


def resize_base64_image(base64_str, target_size=(256, 256)):
    if not base64_str:
        return base64_str
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        resized_image = image.resize(target_size)
        buffer = BytesIO()
        resized_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error resizing image: {e}")
        return base64_str


def retrieve_experience_from_chroma(question):
    try:
        db_file = "/path/to/CG-chroma_db-ALL/chroma_db-CG-DATA-5qwen3plus/chroma.sqlite3"
        db_dir = os.path.dirname(db_file)
        client = chromadb.PersistentClient(path=db_dir)
        try:
            collection = client.get_collection(name="nav_experiences")
        except Exception:
            cols = client.list_collections()
            if not cols:
                return None
            name = cols[0].name if hasattr(cols[0], "name") else cols[0]
            collection = client.get_collection(name=name)
        res = collection.query(
            query_texts=[f"IF answering {question}"],
            where={"type": "eqa"},
            n_results=1,
            include=["documents"],
        )
        docs = res.get("documents")
        if docs and len(docs) > 0 and len(docs[0]) > 0:
            return docs[0][0]
    except Exception:
        return None
    return None


def format_explore_prompt_evolver(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    frontier_classes=None,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    experience=None,
    use_experience=True,
):
    # Resize all input images to 256x256
    target_size = (256, 256)
    if egocentric_imgs:
        egocentric_imgs = [resize_base64_image(img, target_size) for img in egocentric_imgs]
    if frontier_imgs:
        frontier_imgs = [resize_base64_image(img, target_size) for img in frontier_imgs]
    if snapshot_imgs:
        snapshot_imgs = [resize_base64_image(img, target_size) for img in snapshot_imgs]
    if image_goal is not None:
        image_goal = resize_base64_image(image_goal, target_size)
    
    if use_experience and experience is None:
        experience = retrieve_experience_from_chroma(question)

    # Construct prompt intro
    exp_section = ""
    if experience:
        exp_section = (
            "<EXP>\n"
            "Guidance from Memory:\n"
            f"{experience}\n"
            "(Instruction: Carefully check if any candidate image contains the visual cues mentioned in the 'IF' condition of this experience. If a match is found, strictly prioritize that path as per the 'THEN' rule.)\n"
            "</EXP>\n\n"
        )

    sys_prompt = (
        "Task: You are an intelligent robot navigating in an indoor scene. Your task is to select a Frontier Image for further exploration or a Memory Image for answering the given Question.\n\n"
    )

    content = []
    # 1 Question
    text = f"Question:\n{question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n\n",))
    else:
        content.append((text + "\n\n",))

    if exp_section:
        content.append((exp_section,))

    text = (
        "Definitions:\n"
        "1. Frontier Image: An observation of unexplored areas that may provide new clues for answering the Question. Selecting a Frontier Image means that you will further explore that direction. If you choose a Frontier image, you need to explain why you would like to choose that direction to explore.\n"
        "2. Memory Image: An observation of several known objects. Selecting a Memory Image means that you have found the final destination to answer the Question. If you choose a Memory Image, you need to directly give an answer to the question. If you don't have enough information to give an answer, then don't choose a Memory Image.\n"
        "Candidate Images:\n"
    )
    content.append((text,))

    # List Frontier Images
    if frontier_imgs:
        content.append(("\n[Frontier Images]\n",))
        for i, img in enumerate(frontier_imgs):
            content.append((f"Frontier Image {i}: ", img))
            if frontier_classes and i < len(frontier_classes) and frontier_classes[i]:
                labels_text = ", ".join(frontier_classes[i])
                content.append((f"\n[Detected Objects]: {labels_text}\n",))
            else:
                content.append(("\n",))
    else:
        content.append(("\n[Frontier Images]\n",))
        content.append(("\nNo Frontier Images available.\n",))

    # List Memory Images
    if snapshot_imgs:
        content.append(("\n[Memory Images]\n",))
        for i, img in enumerate(snapshot_imgs):
            content.append((f"Memory Image {i}: ", img))
            if use_snapshot_class and i < len(snapshot_classes):
                labels_text = ", ".join(snapshot_classes[i])
                content.append((f"\n[Detected Objects]: {labels_text}\n",))
            else:
                content.append(("\n",))
    else:
        content.append(("\n[Memory Images]\n",))
        content.append(("\nNo Memory Images available.\n",))


    # Instruction for Reasoning
    prompt_instruction = (
        "\nInstruction for Reasoning or Answer:\n"
        f"1. Find visual evidence for \"{question}\" considering <EXP>.\n"
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
    content.append((prompt_instruction,))

    return sys_prompt, content


def format_prefiltering_prompt(question, class_list, top_k=10, image_goal=None):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
    prompt = "Your goal is to answer questions about the scene through exploration.\n"
    prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
    prompt += "These are the rules for the task.\n"
    prompt += "1. Read through the whole object list.\n"
    prompt += "2. Rank objects in the list based on how well they can help your exploration given the question.\n"
    prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
    prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
    content.append((prompt,))
    # ------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects:\n"
    prompt += "Question: What can I use to watch my favorite shows and movies?\n"
    prompt += (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ntv\nbook rack\nsofa\noven\nbed\ncurtain\n"
    prompt += "Answer: tv\nspeaker\nsofa\nbed\n"
    content.append((prompt,))
    # ------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt + "\n",))
    prompt = (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    for i, cls in enumerate(class_list):
        prompt += f"{cls}\n"
    prompt += "Answer: "
    content.append((prompt,))
    return sys_prompt, content


def get_prefiltering_classes(question, seen_classes, top_k=10, image_goal=None):
    prefiltering_sys, prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal
    )

    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image {c[1][:10]}..."
    response = call_openai_api(prefiltering_sys, prefiltering_content)
    if response is None:
        return []

    # parse the response and return the top_k objects
    selected_classes = response.strip().split("\n")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]

    return selected_classes


def prefiltering(
    question, snapshot_classes, seen_classes, top_k=10, image_goal=None, verbose=False
):
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k, image_goal
    )
    if verbose:
        logging.info(f"Prefiltering selected classes: {selected_classes}")

    keep_index = [
        i
        for i in range(len(snapshot_classes))
        if len(set(snapshot_classes[i]) & set(selected_classes)) > 0
    ]
    snapshot_classes = [snapshot_classes[i] for i in keep_index]
    snapshot_classes = [
        sorted(list(set(s_cls) & set(selected_classes))) for s_cls in snapshot_classes
    ]
    return snapshot_classes, keep_index


_CLIP_MODEL = None
_CLIP_PREPROCESS = None

def get_clip_model():
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
    return _CLIP_MODEL, _CLIP_PREPROCESS

def filter_images_by_relevance(question, frontier_imgs, snapshot_imgs, snapshot_classes, snapshot_id_mapping, top_k=4):
    total_imgs = len(frontier_imgs) + len(snapshot_imgs)
    if total_imgs <= top_k:
        return frontier_imgs, snapshot_imgs, snapshot_classes, snapshot_id_mapping

    model, preprocess = get_clip_model()
    device = next(model.parameters()).device

    # Prepare text features
    # Limit text length to 77 tokens as CLIP constraint
    text = clip.tokenize([question[:70]]).to(device) 
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Prepare image features
    scores = []
    
    # Frontiers
    for i, img_b64 in enumerate(frontier_imgs):
        try:
            img = Image.open(BytesIO(base64.b64decode(img_b64)))
            img_input = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_input)
                feat /= feat.norm(dim=-1, keepdim=True)
                score = (feat @ text_features.T).item()
                scores.append((score, 'frontier', i))
        except Exception as e:
            logging.error(f"Error processing frontier image {i}: {e}")
            scores.append((-1.0, 'frontier', i))
            
    # Snapshots
    for i, img_b64 in enumerate(snapshot_imgs):
        try:
            img = Image.open(BytesIO(base64.b64decode(img_b64)))
            img_input = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_input)
                feat /= feat.norm(dim=-1, keepdim=True)
                score = (feat @ text_features.T).item()
                scores.append((score, 'snapshot', i))
        except Exception as e:
            logging.error(f"Error processing snapshot image {i}: {e}")
            scores.append((-1.0, 'snapshot', i))

    # Sort and keep top k
    scores.sort(key=lambda x: x[0], reverse=True)
    top_k_scores = scores[:top_k]
    
    keep_frontier_indices = set()
    keep_snapshot_indices = set()
    
    for _, type_, idx in top_k_scores:
        if type_ == 'frontier':
            keep_frontier_indices.add(idx)
        else:
            keep_snapshot_indices.add(idx)
            
    new_frontier_imgs = [frontier_imgs[i] for i in range(len(frontier_imgs)) if i in keep_frontier_indices]
    
    new_snapshot_imgs = []
    new_snapshot_classes = []
    new_snapshot_id_mapping = []
    
    for i in range(len(snapshot_imgs)):
        if i in keep_snapshot_indices:
            new_snapshot_imgs.append(snapshot_imgs[i])
            new_snapshot_classes.append(snapshot_classes[i])
            new_snapshot_id_mapping.append(snapshot_id_mapping[i])
            
    return new_frontier_imgs, new_snapshot_imgs, new_snapshot_classes, new_snapshot_id_mapping


def explore_step(step, cfg, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        frontier_classes,
        snapshot_imgs,
        snapshot_classes,
        snapshot_id_mapping,
    ) = get_step_info(step, verbose)

    # # Filter images to keep top 4 most relevant to question
    # (
    #     frontier_imgs,
    #     snapshot_imgs,
    #     snapshot_classes,
    #     snapshot_id_mapping
    # ) = filter_images_by_relevance(
    #     question, 
    #     frontier_imgs, 
    #     snapshot_imgs, 
    #     snapshot_classes, 
    #     snapshot_id_mapping,
    #     top_k=4
    # )

    sys_prompt, content = format_explore_prompt_evolver(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        use_experience=getattr(cfg, "use_experience", True),
    )

    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 3
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        full_response = call_openai_api(sys_prompt, content)

        if full_response is None:
            print("call_openai_api returns None, retrying")
            continue

        # Try to parse response from format_explore_prompt_evolver
        # Expected format: 
        # "Frontier Image i\nReason: ..."
        # "Memory Image i\nAnswer: ..."
        
        is_evolver_format = False
        import re
        
        # Check for Memory Image
        memory_match = re.search(r"Memory Image\s+(\d+)", full_response, re.IGNORECASE)
        # Check for Frontier Image
        frontier_match = re.search(r"Frontier Image\s+(\d+)", full_response, re.IGNORECASE)

        logging.info(f"Full response: {full_response}")
        
        if memory_match:
            is_evolver_format = True
            image_idx = int(memory_match.group(1))
            choice_type = "snapshot"
            choice_id = str(image_idx)
            
            # Extract Answer
            answer_match = re.search(r"Answer:\s*(.*)", full_response, re.IGNORECASE | re.DOTALL)
            if answer_match:
                reason = answer_match.group(1).strip()
            else:
                reason = ""
                
        elif frontier_match:
            is_evolver_format = True
            image_idx = int(frontier_match.group(1))
            choice_type = "frontier"
            choice_id = str(image_idx)
            
            # Extract Reason
            reason_match = re.search(r"Reason:\s*(.*)", full_response, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = ""
                
        else:
            # Check for legacy "Selection: Image i" pattern just in case
            selection_match = re.search(r"Selection:\s*Image\s+(\d+)", full_response, re.IGNORECASE)
            if selection_match:
                is_evolver_format = True
                image_idx = int(selection_match.group(1))
                
                num_snapshots = len(snapshot_imgs)
                num_frontiers = len(frontier_imgs)
                
                if 0 <= image_idx < num_snapshots:
                    choice_type = "snapshot"
                    choice_id = str(image_idx)
                    # For legacy format, we might need CoT or just take reason
                    reason_match = re.search(r"Reasoning:\s*(.*)", full_response, re.IGNORECASE)
                    if reason_match:
                        reason = reason_match.group(1).strip()
                    else:
                        reason = ""
                elif num_snapshots <= image_idx < num_snapshots + num_frontiers:
                    choice_type = "frontier"
                    choice_id = str(image_idx - num_snapshots)
                    reason_match = re.search(r"Reasoning:\s*(.*)", full_response, re.IGNORECASE)
                    if reason_match:
                        reason = reason_match.group(1).strip()
                    else:
                        reason = ""
                else:
                    print(f"Image index {image_idx} out of range")
                    continue
            else:
                pass 

        if is_evolver_format:
            response = f"{choice_type} {choice_id}"
        else:
            # Fallback to original parsing logic
            full_response = full_response.strip()
            if "\n" in full_response:
                lines = full_response.split("\n")
                response = lines[0].strip()
                reason = lines[-1].strip()
            else:
                response = full_response.strip()
                reason = ""
            
            response = response.lower()
            try:
                choice_type, choice_id = response.split(" ")
            except Exception as e:
                print(f"Error in splitting response: {response}")
                print(e)
                continue


        response_valid = False
        if (
            choice_type == "snapshot"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(snapshot_imgs)
        ):
            response_valid = True
        elif (
            choice_type == "frontier"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(frontier_imgs)
        ):
            response_valid = True

        if response_valid:
            final_response = response
            final_reason = reason
            break

    return final_response, snapshot_id_mapping, final_reason, len(snapshot_imgs)
