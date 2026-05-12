import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
import chromadb
from src.const import *


# client = OpenAI(
#     base_url=END_POINT,
#     api_key=OPENAI_KEY,
# )

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

# send information to openai
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
        db_file = "/path/to/chroma_db-CG-DATA-5qwen3plus/chroma.sqlite3"
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
            where={"type": "object_goal"},
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
    snapshot_crops=None,
    frontier_classes=None,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    experience=None,
    use_experience=True,
):
    target_size = (256, 256)
    if egocentric_imgs:
        egocentric_imgs = [resize_base64_image(img, target_size) for img in egocentric_imgs]
    if frontier_imgs:
        frontier_imgs = [resize_base64_image(img, target_size) for img in frontier_imgs]
    if snapshot_imgs:
        if isinstance(snapshot_imgs, dict):
            snapshot_imgs = {
                rgb_id: resize_base64_image(img, target_size)
                for rgb_id, img in snapshot_imgs.items()
            }
        else:
            snapshot_imgs = [resize_base64_image(img, target_size) for img in snapshot_imgs]
    if snapshot_crops:
        snapshot_crops = {
            rgb_id: [resize_base64_image(img, target_size) for img in crops]
            for rgb_id, crops in snapshot_crops.items()
        }
    if image_goal is not None:
        image_goal = resize_base64_image(image_goal, target_size)

    if use_experience and experience is None:
        experience = retrieve_experience_from_chroma(question)

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
        "Task: You are an intelligent robot navigating in an indoor scene. Your task is to select a Frontier Image for further exploration or a Memory Image + Object for answering the given Question.\n\n"
    )

    content = []
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
        "1. Frontier Image: An observation of unexplored areas that may provide new clues for answering the Question. Selecting a Frontier Image means that you will further explore that direction. If you choose a Frontier Image, you need to explain why you would like to choose that direction to explore.\n"
        "2. Memory Image: A focused observation of several objects. Selecting a Memory Image means that you have found the relevant area and should choose ONE object within that memory image as the navigation target to answer the Question.\n"
        "Candidate Images:\n"
    )
    content.append((text,))

    # if egocentric_view and egocentric_imgs:
    #     content.append(("\n[Egocentric View]\n",))
    #     content.append(("Current egocentric view: ", egocentric_imgs[-1]))
    #     content.append(("\n",))

    content.append(("\n[Frontier Images]\n",))
    if frontier_imgs:
        for i, img in enumerate(frontier_imgs):
            content.append((f"Frontier Image {i}: ", img))
            if frontier_classes and i < len(frontier_classes) and frontier_classes[i]:
                labels_text = ", ".join(frontier_classes[i])
                content.append((f"\n[Detected Objects]: {labels_text}\n",))
            else:
                content.append(("\n",))
    else:
        content.append(("\nNo Frontier Images available.\n",))

    content.append(("\n[Memory Images]\n",))
    if snapshot_imgs and isinstance(snapshot_imgs, dict):
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            content.append((f"Memory Image {i}: ", snapshot_imgs[rgb_id]))
            if snapshot_crops and rgb_id in snapshot_crops:
                for j in range(len(snapshot_crops[rgb_id])):
                    if use_snapshot_class and rgb_id in snapshot_classes and j < len(
                        snapshot_classes[rgb_id]
                    ):
                        obj_text = f"Object {j}: {snapshot_classes[rgb_id][j]}"
                    else:
                        obj_text = f"Object {j}:"
                    content.append((obj_text, snapshot_crops[rgb_id][j]))
                content.append(("\n",))
            else:
                if use_snapshot_class and rgb_id in snapshot_classes:
                    labels_text = ", ".join(snapshot_classes[rgb_id])
                    content.append((f"\n[Detected Objects]: {labels_text}\n",))
                else:
                    content.append(("\n",))
    else:
        content.append(("\nNo Memory Images available.\n",))

    prompt_instruction = (
        "\nInstruction for Reasoning or Answer:\n"
        f"1. Find visual evidence for \"{question}\" considering <EXP>.\n"
        "2. Response constraints:\n"
        "   - The Reason or Answer must be a simple, direct, natural sentence understandable by others. NO meta-words (e.g., 'this image').\n"
        "   - You should always choose the ONE most relevant Memory/Frontier index (and ONE object index if choosing a Memory Image).\n"
        "3. Provide your response in the following strict format:\n"
        "   - If you choose a Frontier Image (for reasoning/exploration), return:\n"
        "     \"Frontier Image i\\nReason: [Reason]\"\n"
        "   - If you choose a Memory Image (for answering), return:\n"
        "     \"Memory Image i, Object j\\nAnswer: [Answer]\"\n"
        "   (Where i is the chosen image index and j is the object index within the chosen Memory Image)\n"
        "4. Examples:\n"
        "Frontier Image 0\nReason: The hallway likely leads to the living room.\n"
        "Memory Image 1, Object 2\nAnswer: The mug is on the kitchen counter.\n"
        "Now return your response\n"
    )
    content.append((prompt_instruction,))

    return sys_prompt, content


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

    # 2.3 get snapshots
    snapshot_classes = {}  # rgb_id -> list of classes
    snapshot_full_imgs = {}  # rgb_id -> full img
    snapshot_crops = {}  # rgb_id -> list of crops
    snapshot_clusters = {}  # rgb_id -> list of clusters
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]["full_img"]
        snapshot_full_imgs[rgb_id] = encode_tensor2base64(snapshot_img)
        snapshot_crops[rgb_id] = [
            encode_tensor2base64(crop_data["crop"])
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        snapshot_class = [
            crop_data["obj_class"]
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        cluster_class = [
            obj_map[int(obj_id)] for obj_id in step["snapshot_objects"][rgb_id]
        ]
        # remove duplicates
        seen_classes.update(sorted(list(set(snapshot_class))))
        snapshot_classes[rgb_id] = snapshot_class
        snapshot_clusters[rgb_id] = cluster_class

    # 3 prefiltering, note that we need the obj_id_mapping
    keep_index = list(range(len(snapshot_full_imgs)))
    keep_index_snapshot = {
        rgb_id: list(range(len(snapshot_crops[rgb_id]))) for rgb_id in snapshot_crops
    }
    if step.get("use_prefiltering") is True:
        use_full_obj_list = step["use_full_obj_list"]
        n_prev_snapshot = len(snapshot_full_imgs)
        snapshot_classes, keep_index, keep_index_snapshot = prefiltering(
            question,
            snapshot_classes,
            snapshot_clusters,
            seen_classes,
            step["top_k_categories"],
            image_goal,
            use_full_obj_list,
            verbose=verbose,
        )
        snapshot_full_imgs = {
            rgb_id: snapshot_full_imgs[rgb_id] for rgb_id in keep_index_snapshot.keys()
        }
        for rgb_id in snapshot_classes.keys():
            snapshot_crops[rgb_id] = [
                snapshot_crops[rgb_id][i] for i in keep_index_snapshot[rgb_id]
            ]
        if verbose:
            logging.info(
                f"Prefiltering snapshot: {n_prev_snapshot} -> {len(snapshot_full_imgs)}"
            )

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        keep_index,
        keep_index_snapshot,
    )


def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    snapshot_crops,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene that is able to observe the surroundings and explore the environment. You are tasked with indoor navigation, and you are required to choose either a Snapshot or a Frontier image to explore and find the target object required in the question.\n"

    content = []
    # 1 here is some basic info
    text = "Definitions:\n"
    text += (
        "Snapshot: A focused observation of several objects. It contains a full image of the cluster of objects, and separate image crops of each object. "
        + "Choosing a snapshot means that the object asked in the question is within the cluster of objects that the snapshot represents, and you will choose that object as the final answer of the question. "
        + "Therefore, if you choose a snapshot, you should also choose the object in the snapshot that you think is the answer to the question.\n"
    )
    text += "Frontier: An unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction.\n"

    # 2 here is the question
    text += f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Frontier/Snapshot that would help find the answer of the question.\n"
    content.append((text,))

    # 3 here is the egocentric views
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))

    # 4 here is the snapshot images
    text = "The followings are all the snapshots that you can choose. Following each snapshot image are the class name and image crop of each object contained in the snapshot.\n"
    text += "Please note that the class name may not be accurate due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make the decision.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            content.append((f"Snapshot {i} ", snapshot_imgs[rgb_id]))
            for j in range(len(snapshot_crops[rgb_id])):
                content.append(
                    (
                        f"Object {j}: {snapshot_classes[rgb_id][j]}",
                        snapshot_crops[rgb_id][j],
                    )
                )
            content.append(("\n",))

    # 5 here is the frontier images
    text = "The followings are all the Frontiers that you can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))

    # 6 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i, Object j' or 'Frontier i', where i, j are the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the fridge in the first snapshot, please return 'Snapshot 0, Object 2', where 2 is the index of the fridge in that snapshot.\n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    content.append((text,))

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
    question,
    snapshot_classes,
    snapshot_clusters,
    seen_classes,
    top_k=10,
    image_goal=None,
    use_full_obj_list=False,
    verbose=False,
):
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k, image_goal
    )
    if verbose:
        logging.info(f"Prefiltering selected classes: {selected_classes}")

    keep_index = [
        i
        for i, k in enumerate(snapshot_clusters.keys())
        if len(set(snapshot_clusters[k]) & set(selected_classes)) > 0
    ]
    keep_snapshot_id = [list(snapshot_classes.keys())[i] for i in keep_index]
    snapshot_classes = {rgb_id: snapshot_classes[rgb_id] for rgb_id in keep_snapshot_id}

    keep_index_snapshot = {}
    for rgb_id in keep_snapshot_id:
        keep_index_snapshot[rgb_id] = [
            i
            for i in range(len(snapshot_classes[rgb_id]))
            if snapshot_classes[rgb_id][i] in selected_classes
        ]
        snapshot_classes[rgb_id] = [
            snapshot_classes[rgb_id][i] for i in keep_index_snapshot[rgb_id]
        ]

    return snapshot_classes, keep_index, keep_index_snapshot


def explore_step(step, cfg, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
    ) = get_step_info(step, verbose)
    sys_prompt, content = format_explore_prompt_evolver(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        frontier_classes=None,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        use_experience=getattr(cfg, "use_experience", True),
    )

    # sys_prompt, content = format_explore_prompt(
    #     question,
    #     egocentric_imgs,
    #     frontier_imgs,
    #     snapshot_full_imgs,
    #     snapshot_classes,
    #     snapshot_crops,
    #     egocentric_view=step.get("use_egocentric_views", False),
    #     use_snapshot_class=True,
    #     image_goal=image_goal,
    # )

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

        full_response = full_response.strip()
        is_evolver_format = False
        import re

        snapshot_match = re.search(
            r"(?:Snapshot|Memory Image)\s+(\d+)", full_response, re.IGNORECASE
        )
        object_match = re.search(r"Object\s+(\d+)", full_response, re.IGNORECASE)
        frontier_match = re.search(
            r"Frontier(?:\s+Image)?\s+(\d+)", full_response, re.IGNORECASE
        )

        if snapshot_match and object_match:
            is_evolver_format = True
            image_idx = int(snapshot_match.group(1))
            obj_idx = int(object_match.group(1))
            response = f"snapshot {image_idx}, object {obj_idx}"
            answer_match = re.search(
                r"Answer:\s*(.*)", full_response, re.IGNORECASE | re.DOTALL
            )
            if answer_match:
                reason = answer_match.group(1).strip()
            else:
                reason_match = re.search(
                    r"Reason:\s*(.*)", full_response, re.IGNORECASE | re.DOTALL
                )
                reason = reason_match.group(1).strip() if reason_match else ""
        elif frontier_match:
            is_evolver_format = True
            image_idx = int(frontier_match.group(1))
            response = f"frontier {image_idx}"
            reason_match = re.search(
                r"Reason:\s*(.*)", full_response, re.IGNORECASE | re.DOTALL
            )
            reason = reason_match.group(1).strip() if reason_match else ""
        else:
            response = full_response
            if "\n" in response:
                response = response.split("\n")
                response, reason = response[0], response[-1]
            else:
                reason = ""

        response = response.lower()
        try:
            choice_type, choice_id = response.split(",")[0].strip().split(" ")
        except Exception as e:
            print(f"Error in splitting response: {response}")
            print(e)
            continue

        response_valid = False
        if (
            choice_type == "snapshot"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(snapshot_full_imgs)
        ):
            try:
                object_choice_type, object_choice_id = (
                    response.split(",")[1].strip().split(" ")
                )
            except Exception as e:
                print(f"Error in splitting response: {response}")
                print(e)
                continue
            if (
                object_choice_type == "object"
                and object_choice_id.isdigit()
                and 0
                <= int(object_choice_id)
                < len(list(snapshot_crop_mapping.values())[int(choice_id)])
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

    return (
        final_response,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        final_reason,
        len(snapshot_full_imgs),
    )
