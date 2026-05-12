import argparse
import json
import os
import pickle
import time
from statistics import mean

from omegaconf import OmegaConf
from openai import OpenAI

prompt_template = '''
You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer or any of the extra answers.
1 means that the response is completely different from the answer and all of the extra answers.

Example 1:
Question: Is it overcast?
Answer: no
Extra Answers: ['doesn't look like it', 'no',' it's sunny']
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Extra Answers: ['a woman', 'a lady', 'woman']
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Extra Answers: {extra_answers}
Response: {prediction}
'''



def create_qwen_client():
    api_key = "xxx"
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is not set in environment variables")
    base_url="xxx"

    return OpenAI(api_key=api_key, base_url=base_url)


def call_qwen_score(client, question, gt_answer, pred_answer, max_retries=5):
    system_prompt = prompt_template
    user_prompt = (
        f"Question: {question}\n"
        f"Ground truth answer: {gt_answer}\n"
        f"Predicted answer: {pred_answer}\n\n"
        "Score (1-5):"
    )
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="xxx",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=8,
                top_p=1.0,
            )
            content = completion.choices[0].message.content.strip()
            score = int(content.split()[0])
            if score < 1 or score > 5:
                raise ValueError("Score out of range")
            return float(score)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2.0)


def load_questions(questions_file, eval_cfg_path=None):
    if questions_file:
        path = questions_file
    elif eval_cfg_path and os.path.exists(eval_cfg_path):
        cfg = OmegaConf.load(eval_cfg_path)
        OmegaConf.resolve(cfg)
        path = cfg.questions_list_path
    else:
        path = "data/aeqa_questions-184.json"
    with open(path, "r") as f:
        questions = json.load(f)
    qid_to_item = {item["question_id"]: item for item in questions}
    return qid_to_item


def load_path_lengths(exp_dir):
    agg_path = os.path.join(exp_dir, "path_length_list.pkl")
    if os.path.exists(agg_path):
        target_path = agg_path
    else:
        candidates = [
            p
            for p in os.listdir(exp_dir)
            if p.startswith("path_length_list_") and p.endswith(".pkl")
        ]
        if not candidates:
            return {}
        candidates.sort()
        target_path = os.path.join(exp_dir, candidates[0])
    with open(target_path, "rb") as f:
        data = pickle.load(f)
    return data


def compute_length_weight(path_lengths):
    if not path_lengths:
        return {}
    lengths = list(path_lengths.values())
    l_min = min(lengths)
    l_max = max(lengths)
    if l_max == l_min:
        return {k: 1.0 for k in path_lengths}
    weights = {}
    for k, v in path_lengths.items():
        w = (l_max - float(v)) / (l_max - l_min)
        if w < 0.0:
            w = 0.0
        if w > 1.0:
            w = 1.0
        weights[k] = w
    return weights


def evaluate_llm_match(
    exp_dir,
    gpt_answer_file,
    questions_file=None,
    eval_cfg_path=None,
    output_file=None,
):
    client = create_qwen_client()
    with open(gpt_answer_file, "r") as f:
        pred_list = json.load(f)
    qid_to_item = load_questions(questions_file, eval_cfg_path)
    path_lengths = load_path_lengths(exp_dir)
    length_weights = compute_length_weight(path_lengths)
    per_question = []
    raw_scores = []
    match_scores = []
    spl_scores = []
    category_raw_scores = {}
    category_match_scores = {}
    category_spl_scores = {}
    for entry in pred_list:
        qid = entry["question_id"]
        pred_answer = entry.get("answer", "")
        question_item = qid_to_item.get(qid)
        if question_item is None:
            continue
        question = question_item["question"]
        gt_answer = question_item["answer"]
        category = question_item.get("category", "unknown")
        raw_score = call_qwen_score(client, question, gt_answer, pred_answer)

        match_score = (raw_score - 1.0) / 4.0 * 100.0 #####

        path_length = float(path_lengths.get(qid, 0.0))
        weight = float(length_weights.get(qid, 0.0))
        spl_score = match_score * weight
        per_question.append(
            {
                "question_id": qid,
                "question": question,
                "category": category,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "raw_score_1_5": raw_score,
                "llm_match": match_score,
                "path_length": path_length,
                "length_weight": weight,
                "llm_match_spl": spl_score,
            }
        )
        raw_scores.append(raw_score)
        match_scores.append(match_score)
        spl_scores.append(spl_score)
        if category not in category_raw_scores:
            category_raw_scores[category] = []
            category_match_scores[category] = []
            category_spl_scores[category] = []
        category_raw_scores[category].append(raw_score)
        category_match_scores[category].append(match_score)
        category_spl_scores[category].append(spl_score)
    num_samples = len(per_question)
    total_samples = len(qid_to_item)
    sum_raw = sum(raw_scores)
    sum_match = sum(match_scores)
    sum_spl = sum(spl_scores)
    overall = {
        "num_samples": num_samples,
        "total_samples": total_samples,
        "mean_raw_score_1_5": sum_raw / num_samples if num_samples > 0 else 0.0,
        "mean_llm_match": sum_match / num_samples if num_samples > 0 else 0.0,
        "mean_llm_match_spl": sum_spl / num_samples if num_samples > 0 else 0.0,
        "weighted_mean_raw_score_1_5": (
            (sum_raw + (total_samples - num_samples) * 1.0) / total_samples
            if total_samples > 0
            else 0.0
        ),
        "weighted_mean_llm_match": (
            (sum_match + (total_samples - num_samples) * 30.0) / total_samples
            if total_samples > 0
            else 0.0
        ),
        "weighted_mean_llm_match_spl": (
            sum_spl / total_samples if total_samples > 0 else 0.0
        ),
    }
    category_total_counts = {}
    for item in qid_to_item.values():
        cat = item.get("category", "unknown")
        category_total_counts[cat] = category_total_counts.get(cat, 0) + 1
    category_overall = {}
    for cat in category_raw_scores:
        scores_raw = category_raw_scores[cat]
        scores_match = category_match_scores[cat]
        scores_spl = category_spl_scores[cat]
        num_cat = len(scores_raw)
        total_cat = category_total_counts.get(cat, num_cat)
        sum_raw_cat = sum(scores_raw)
        sum_match_cat = sum(scores_match)
        sum_spl_cat = sum(scores_spl)
        frac_cat = total_cat / total_samples if total_samples > 0 else 0.0
        mean_raw_cat = sum_raw_cat / num_cat if num_cat > 0 else 0.0
        mean_match_cat = sum_match_cat / num_cat if num_cat > 0 else 0.0
        mean_spl_cat = sum_spl_cat / num_cat if num_cat > 0 else 0.0
        weighted_mean_raw_cat = (
            (sum_raw_cat + (total_cat - num_cat) * 1.0) / total_cat
            if total_cat > 0
            else 0.0
        )
        weighted_mean_match_cat = (
            (sum_match_cat + (total_cat - num_cat) * 30.0) / total_cat
            if total_cat > 0
            else 0.0
        )
        weighted_mean_spl_cat = (
            sum_spl_cat / total_cat if total_cat > 0 else 0.0
        )
        category_overall[cat] = {
            "num_samples": num_cat,
            "total_samples": total_cat,
            "dataset_fraction": frac_cat,
            "mean_raw_score_1_5": mean_raw_cat,
            "mean_llm_match": mean_match_cat,
            "mean_llm_match_spl": mean_spl_cat,
            "weighted_mean_raw_score_1_5": weighted_mean_raw_cat,
            "weighted_mean_llm_match": weighted_mean_match_cat,
            "weighted_mean_llm_match_spl": weighted_mean_spl_cat,
            "global_contrib_mean_raw_score_1_5": mean_raw_cat * frac_cat,
            "global_contrib_mean_llm_match": mean_match_cat * frac_cat,
            "global_contrib_mean_llm_match_spl": mean_spl_cat * frac_cat,
            "global_contrib_weighted_mean_raw_score_1_5": weighted_mean_raw_cat
            * frac_cat,
            "global_contrib_weighted_mean_llm_match": weighted_mean_match_cat
            * frac_cat,
            "global_contrib_weighted_mean_llm_match_spl": weighted_mean_spl_cat
            * frac_cat,
        }
    result = {
        "per_question": per_question,
        "overall": overall,
        "category_overall": category_overall,
    }
    if output_file is None:
        output_file = os.path.join(exp_dir, "llm_match_results.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results/xxx",
    )
    parser.add_argument(
        "--gpt_answer_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval_cfg",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.exp_dir
    
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return

    # Determine directories to process
    # If root_dir contains the target file, process it directly.
    # Otherwise, check its subdirectories.
    target_dirs = []
    default_gpt_file = "gpt_answer_0.0_1.0.json"
    
    if os.path.exists(os.path.join(root_dir, default_gpt_file)):
        target_dirs.append(root_dir)
    else:
        # Check subdirectories
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for subdir in subdirs:
            full_path = os.path.join(root_dir, subdir)
            if os.path.exists(os.path.join(full_path, default_gpt_file)):
                target_dirs.append(full_path)
    
    if not target_dirs:
        print(f"No directories containing '{default_gpt_file}' found in or under {root_dir}")
        return

    for exp_dir in target_dirs:
        print(f"Processing {exp_dir}...")
        
        # Determine gpt_answer_file
        if args.gpt_answer_file and len(target_dirs) == 1:
            gpt_answer_file = args.gpt_answer_file
        else:
            gpt_answer_file = os.path.join(exp_dir, default_gpt_file)
        
        if not os.path.exists(gpt_answer_file):
            print(f"  Skipping {exp_dir}: {gpt_answer_file} not found.")
            continue

        # Determine eval_cfg_path
        if args.eval_cfg and len(target_dirs) == 1:
            eval_cfg_path = args.eval_cfg
        else:
            eval_cfg_path = os.path.join(exp_dir, "eval_aeqa_184_train.yaml")
        
        try:
            evaluate_llm_match(
                exp_dir=exp_dir,
                gpt_answer_file=gpt_answer_file,
                questions_file=args.questions_file,
                eval_cfg_path=eval_cfg_path,
                output_file=None,  # Always save to default location inside exp_dir
            )
        except Exception as e:
            print(f"  Error processing {exp_dir}: {e}")


if __name__ == "__main__":
    main()
