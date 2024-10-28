import json

eval_path = r"C:\Users\86159\Desktop\subgraph_kgp1_valid.json"
ground_truth_path = r"C:\Users\86159\Desktop\subgraph_kgp1_valid.json"

# 使用 'utf-8' 编码来读取文件
with open(eval_path, 'r', encoding='utf-8') as file:
    eval_dict = json.load(file)

with open(ground_truth_path, 'r', encoding='utf-8') as file:
    label_dict = json.load(file)

for task in ['link_prediction', 'entity_prediction']:
    evals = eval_dict.get(task, [])
    labels = label_dict.get(task, [])
    total_scores = 0
    for eval, label in zip(evals, labels):
        eval_output = eval.get("output", [])
        label_truth = set(label.get("ground_truth", []))
        for idx, output in enumerate(eval_output):
            if output in label_truth:
                total_scores += 1.0 / (idx + 1)

    # 确保分母不为零
    average_score = total_scores / len(evals) if evals else 0
    print(task, average_score)


