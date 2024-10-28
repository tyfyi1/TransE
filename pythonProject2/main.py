import pandas as pd
import torch

file_path = r'C:\Users\86159\Desktop\workreport\subgraph_kgp1.txt'

rows = []

with open(file_path, encoding='utf-8') as file:
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

print(df)

unique_entities = set(df['start_entity']).union(set(df['end_entity']))
unique_relations = set(df['relation'])

entity_to_id = {entity: idx for idx, entity in enumerate(unique_entities)}
relation_to_id = {relation: idx for idx, relation in enumerate(unique_relations)}

df['start_entity_id'] = df['start_entity'].map(entity_to_id)
df['relation_id'] = df['relation'].map(relation_to_id)
df['end_entity_id'] = df['end_entity'].map(entity_to_id)

print(df[['start_entity', 'start_entity_id', 'relation', 'relation_id', 'end_entity', 'end_entity_id']])

entity_count = len(unique_entities)
relation_count = len(unique_relations)

print(f"实体数量: {entity_count}")
print(f"关系数量: {relation_count}")

triplets = df[['start_entity', 'relation', 'end_entity']].values.tolist()

print(triplets[:5])
relation_dict = {}

for start_entity, relation, end_entity in triplets:
    if relation not in relation_dict:
        relation_dict[relation] = []
    relation_dict[relation].append((start_entity, end_entity))

for relation, entity_pairs in list(relation_dict.items())[:5]:
    print(f"Relation: {relation}, Entity Pairs: {entity_pairs}")

