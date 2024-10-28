import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # 导入numpy库
import matplotlib.pyplot as plt
import random

ENTITY_NUM = 162336
RELATION_NUM = 47
EMBEDDING_DIM = 5
LR = 0.1
BATCH_SIZE = 256

import pandas as pd


file_path = r'C:\Users\86159\Desktop\workreport\subgraph_kgp1.txt'
rows = []
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

unique_entities = set(df['start_entity']).union(set(df['end_entity']))
unique_relations = set(df['relation'])

entity_to_id = {entity: idx for idx, entity in enumerate(unique_entities)}
relation_to_id = {relation: idx for idx, relation in enumerate(unique_relations)}

df['start_entity_id'] = df['start_entity'].map(entity_to_id)
df['relation_id'] = df['relation'].map(relation_to_id)
df['end_entity_id'] = df['end_entity'].map(entity_to_id)



entity_count = len(unique_entities)
relation_count = len(unique_relations)

triplets = torch.tensor(df[['start_entity_id', 'relation_id', 'end_entity_id']].values)


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

        score = head_embeddings + relation_embeddings - tail_embeddings
        return score

    def distance(self, heads, relations, tails):
        score = self.forward(heads, relations, tails)
        return torch.norm(score, p=1, dim=1)


model = TransE(ENTITY_NUM, RELATION_NUM, EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=LR)
model_path = r'D:\pycode\pythonProject2\model\model.pth'
model.load_state_dict(torch.load(model_path))
optimizer_path=r'D:\pycode\pythonProject2\optimizer\optimizer.pth'
optimizer.load_state_dict(torch.load(optimizer_path, weights_only=False))
# 训练模型

train_losses = []
eval_losses = []

# 负样本生成函数
def generate_negative_samples(triplets, entity_count, negative_sample_size=1):
    negative_triplets = []
    for triplet in triplets:
        head, relation, tail = triplet
        for _ in range(negative_sample_size):
            # 随机替换头实体或尾实体
            if random.random() < 0.5:
                # 替换头实体
                negative_head = random.randint(0, entity_count - 1)
                while negative_head == head:
                    negative_head = random.randint(0, entity_count - 1)
                negative_triplets.append([negative_head, relation, tail])
            else:
                # 替换尾实体
                negative_tail = random.randint(0, entity_count - 1)
                while negative_tail == tail:
                    negative_tail = random.randint(0, entity_count - 1)
                negative_triplets.append([head, relation, negative_tail])
    return torch.tensor(negative_triplets)


# 评估函数
def evaluate(model, triplets, entity_count, batch_size=128):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i in range(0, len(triplets), batch_size):
            batch_triplets = triplets[i:i + batch_size]
            heads, relations, tails = batch_triplets[:, 0], batch_triplets[:, 1], batch_triplets[:, 2]

            # 生成负样本
            negative_samples = generate_negative_samples(batch_triplets, entity_count, negative_sample_size=1)
            negative_heads, negative_relations, negative_tails = negative_samples[:, 0], negative_samples[:,
                                                                                         1], negative_samples[:, 2]

            # 计算正样本和负样本的损失
            positive_loss = model.distance(heads, relations, tails).mean()
            negative_loss = model.distance(negative_heads, negative_relations, negative_tails).mean()

            # 计算总损失
            loss = positive_loss + negative_loss
            total_loss += loss.item()
            count += 1
    return total_loss / count

optimizer_path=r'D:\pycode\pythonProject2\optimizer\optimizer.pth'
# 在训练循环中添加负样本生成
for epoch in range(1000):  # 模拟训练1000个epoch
    for i in range(0, len(triplets), BATCH_SIZE):
        batch_triplets = triplets[i:i + BATCH_SIZE]
        heads, relations, tails = batch_triplets[:, 0], batch_triplets[:, 1], batch_triplets[:, 2]

        # 生成负样本
        negative_samples = generate_negative_samples(batch_triplets, ENTITY_NUM, negative_sample_size=1)
        negative_heads, negative_relations, negative_tails = negative_samples[:, 0], negative_samples[:,1], negative_samples[:, 2]

        optimizer.zero_grad()

        # 计算正样本和负样本的损失
        positive_loss = model.distance(heads, relations, tails).mean()
        negative_loss = model.distance(negative_heads, negative_relations, negative_tails).mean()

        # 计算总损失
        loss = positive_loss + negative_loss
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        train_losses.append(loss.item())
        print(f'Epoch {epoch}: Loss = {loss.item()}')

    # 每隔一定epoch进行评估
    if epoch % 10 == 0:
        eval_loss = evaluate(model, triplets, ENTITY_NUM)
        eval_losses.append(eval_loss)  # 记录每个epoch的评估损失
        print(f'Epoch {epoch}: Evaluation Loss = {eval_loss}')

    if epoch % 10 == 0:
        model_path = r'D:\pycode\pythonProject2\model\model.pth'  # 模型保存路径
        optimizer_path = r'D:\pycode\pythonProject2\optimizer\optimizer.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        plt.figure(figsize=(12, 5))

        # 绘制训练损失
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch')
        plt.legend()
        plt.grid(True)

        # 绘制评估损失
        plt.subplot(1, 2, 2)
        plt.plot(eval_losses, label='Evaluation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Evaluation Loss per Epoch')
        plt.legend()
        plt.grid(True)

        # 显示图表
        plt.tight_layout()
        plt.show()
# 保存模型
model_path = r'D:\pycode\pythonProject2\model\model.pth'  # 模型保存路径
optimizer_path=r'D:\pycode\pythonProject2\optimizer'
torch.save(model.state_dict(), model_path)
torch.save(optimizer.state_dict(), optimizer_path)

# 绘制损失折线图
plt.figure(figsize=(12, 5))

# 绘制训练损失
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)

# 绘制评估损失
plt.subplot(1, 2, 2)
plt.plot(eval_losses, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evaluation Loss per Epoch')
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()