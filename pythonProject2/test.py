import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json


class TransE(nn.Module):
    def __init__(self, entity_num, relation_num, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(entity_num, embedding_dim)
        self.relation_embeddings = nn.Embedding(relation_num, embedding_dim)

        # 初始化权重
        nn.init.uniform_(self.entity_embeddings.weight, -6 / np.sqrt(embedding_dim), 6 / np.sqrt(embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight, -6 / np.sqrt(embedding_dim), 6 / np.sqrt(embedding_dim))

    def forward(self, heads, relations, tails):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        # 计算头实体向量 + 关系向量 与 尾实体向量之间的距离
        score = head_embeddings + relation_embeddings - tail_embeddings
        return score

    def distance(self, heads, relations, tails):
        score = self.forward(heads, relations, tails)
        return torch.norm(score, p=1, dim=1)  # 使用L1范数作为距离

rows=[]
file_path = r'C:\Users\86159\Desktop\workreport\subgraph_kgp1.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 去除行尾的换行符，并按空格分割字符串
        parts = line.strip().split()
        # 确保分割后的部分数量正确
        if len(parts) == 12:
            # 将分割后的字符串转换为适当的类型
            row = [
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4],
                parts[5],  # 头实体
                parts[6],  # 关系
                parts[7],  # 尾实体
                int(parts[8]),
                int(parts[9]),
                int(parts[10]),
                int(parts[11])
            ]
            if parts[3] == 'zh':
                rows.append(row)
        else:
            # 如果行的部分数量不正确，可以打印一条错误消息或采取其他措施
            print(f"Skipping malformed line: {line.strip()}")

# 将行列表转换为Pandas DataFrame
df = pd.DataFrame(rows, columns=['ID', 'orgin_id', 'start_lang', 'eng_lang', 'weight', 'start_entity', 'relation', 'end_entity', '-1', '-1', '-1', '-1'])

triplets=df[['start_entity', 'relation', 'end_entity']].values.tolist()


entity_dict = {}
relation_dict = {}
entity_counter = 0
relation_counter = 0


for start_entity, relation, end_entity in triplets:
    # 添加头实体
    if start_entity not in entity_dict:
        entity_dict[start_entity] = entity_counter
        entity_counter += 1
    # 添加尾实体
    if end_entity not in entity_dict:
        entity_dict[end_entity] = entity_counter
        entity_counter += 1
    # 添加关系
    if relation not in relation_dict:
        relation_dict[relation] = relation_counter
        relation_counter += 1


def predict_entity(model, head_entity, relation, entity_dict, relation_dict, top_k=5):
    # 将实体和关系字符串转换为索引
    head_idx = torch.tensor([entity_dict[head_entity]])
    relation_idx = torch.tensor([relation_dict[relation]])

    # 计算头实体和关系的嵌入
    head_embedding = model.entity_embeddings(head_idx)
    relation_embedding = model.relation_embeddings(relation_idx)

    # 计算头实体 + 关系嵌入
    combined_embedding = head_embedding + relation_embedding

    # 计算所有实体与组合嵌入的得分
    all_entity_embeddings = model.entity_embeddings.weight
    scores = -torch.norm(combined_embedding - all_entity_embeddings, p=1, dim=1)

    # 获取得分最高的 top_k 个实体
    top_scores, top_indices = torch.topk(scores, top_k)
    top_entities = [list(entity_dict.keys())[idx] for idx in top_indices]

    return top_entities

model = TransE(162336, 48, embedding_dim=5)
model.load_state_dict(torch.load(r"D:\pycode\pythonProject2\model\model.pth"))

with open(r"C:\Users\86159\Desktop\123.json", 'r',encoding='utf-8') as file:
    data = json.load(file)


# 预测函数，使用上面定义的 predict_entity 函数
def predict_relation(model, head_entity, tail_entity, entity_dict, relation_dict, top_k=5):
    # 将实体字符串转换为索引
    head_idx = torch.tensor([entity_dict[head_entity]])
    tail_idx = torch.tensor([entity_dict[tail_entity]])

    # 计算头实体和尾实体的嵌入
    head_embedding = model.entity_embeddings(head_idx)
    tail_embedding = model.entity_embeddings(tail_idx)

    # 计算头实体嵌入与所有关系嵌入加上尾实体嵌入的得分
    all_relation_embeddings = model.relation_embeddings.weight
    scores = -torch.norm(head_embedding + all_relation_embeddings - tail_embedding, p=1, dim=1)

    # 获取得分最高的 top_k 个关系
    top_scores, top_indices = torch.topk(scores, top_k)
    top_relations = [list(relation_dict.keys())[idx] for idx in top_indices]

    return top_relations


# 遍历 link_prediction 任务并更新 output 字段
for item in data['link_prediction']:
    head_entity = item['input'][0]
    tail_entity = item['input'][1]
    predicted_relations = predict_relation(model, head_entity, tail_entity, entity_dict, relation_dict)
    item['output'] = predicted_relations

# 将更新后的数据写回 JSON 文件
with open(r"C:\Users\86159\Desktop\123.json", 'w',encoding='utf-8') as file:
    json.dump(data, file, indent=4)

print('JSON file has been updated with predictions.')
