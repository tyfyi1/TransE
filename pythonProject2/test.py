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

        nn.init.uniform_(self.entity_embeddings.weight, -6 / np.sqrt(embedding_dim), 6 / np.sqrt(embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight, -6 / np.sqrt(embedding_dim), 6 / np.sqrt(embedding_dim))

    def forward(self, heads, relations, tails):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        score = head_embeddings + relation_embeddings - tail_embeddings
        return score

    def distance(self, heads, relations, tails):
        score = self.forward(heads, relations, tails)
        return torch.norm(score, p=1, dim=1) 

rows=[]
file_path = r'C:\Users\86159\Desktop\workreport\subgraph_kgp1.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 12:
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
            print(f"Skipping malformed line: {line.strip()}")

df = pd.DataFrame(rows, columns=['ID', 'orgin_id', 'start_lang', 'eng_lang', 'weight', 'start_entity', 'relation', 'end_entity', '-1', '-1', '-1', '-1'])

triplets=df[['start_entity', 'relation', 'end_entity']].values.tolist()


entity_dict = {}
relation_dict = {}
entity_counter = 0
relation_counter = 0


for start_entity, relation, end_entity in triplets:
    if start_entity not in entity_dict:
        entity_dict[start_entity] = entity_counter
        entity_counter += 1
    if end_entity not in entity_dict:
        entity_dict[end_entity] = entity_counter
        entity_counter += 1
    if relation not in relation_dict:
        relation_dict[relation] = relation_counter
        relation_counter += 1

entity_dict = {entity: idx for idx, entity in enumerate(set([e for e, _, _ in triplets] + [e for _, _, e in triplets]))}
relation_dict = {relation: idx for idx, relation in enumerate(set([r for _, r, _ in triplets]))}
id_to_entity = {v: k for k, v in entity_dict.items()}
id_to_relation = {v: k for k, v in relation_dict.items()}

model = TransE(162336, 47, embedding_dim=5)
model.load_state_dict(torch.load(r"D:\pycode\pythonProject2\model\model.pth"))

def predict_entity(model, head_entity, relation, entity_dict, relation_dict, id_to_entity, top_k=5):
    head_idx = torch.tensor([entity_dict[head_entity]])
    relation_idx = torch.tensor([relation_dict[relation]])

    head_embedding = model.entity_embeddings(head_idx)
    relation_embedding = model.relation_embeddings(relation_idx)
    combined_embedding = head_embedding + relation_embedding

    all_entity_embeddings = model.entity_embeddings.weight
    scores = -torch.norm(combined_embedding - all_entity_embeddings, p=1, dim=1)

    scores[entity_dict[head_entity]] = float('-inf')

    top_scores, top_indices = torch.topk(scores, top_k)
    top_entities = [id_to_entity[idx.item()] for idx in top_indices]
    return top_entities


def predict_relation(model, head_entity, tail_entity, entity_dict, relation_dict,id_to_relation, top_k=5):
    head_idx = torch.tensor([entity_dict[head_entity]])
    tail_idx = torch.tensor([entity_dict[tail_entity]])

    head_embedding = model.entity_embeddings(head_idx)
    tail_embedding = model.entity_embeddings(tail_idx)

    all_relation_embeddings = model.relation_embeddings.weight
    scores = -torch.norm(head_embedding + all_relation_embeddings - tail_embedding, p=1, dim=1)

    top_scores, top_indices = torch.topk(scores, top_k)

    top_relations = [id_to_relation[idx.item()] for idx in top_indices]
    return top_relations


with open(r"C:\Users\86159\Desktop\123.json",'r',encoding='utf-8') as file:
    data = json.load(file)

for item in data.get('entity_prediction', []):
    head_entity = item['input'][0]
    relation = item['input'][1]
    predicted_entities = predict_entity(model, head_entity, relation, entity_dict, relation_dict,id_to_entity, top_k=5)
    item['output'] = predicted_entities

for item in data.get('link_prediction', []):
    head_entity = item['input'][0]
    tail_entity = item['input'][1]
    predicted_relations = predict_relation(model, head_entity, tail_entity, entity_dict, relation_dict,id_to_relation,top_k=5)
    item['output'] = predicted_relations
with open(r"C:\Users\86159\Desktop\123.json",'w',encoding='utf-8') as file:
    json.dump(data, file,ensure_ascii=False, indent=4)

print('JSON file has been updated with predictions.')
