# TransE
TransE代码
TransE是一种简单且有效的知识图谱嵌入（Knowledge Graph Embedding, KGE）方法，用于将知识图谱中的实体和关系映射到连续的向量空间中。其核心思想是通过学习实体和关系的低维向量表示，使得原始知识图谱中的三元组（头实体、关系、尾实体）在嵌入空间中保持一定的几何关系。

以下是TransE的基本原理：
嵌入表示
实体嵌入：TransE为知识图谱中的每个实体分配一个低维向量。
关系嵌入：同样地，每个关系也被表示为一个低维向量。
目标函数
TransE的目标是使得每个三元组（h, r, t）在嵌入空间中满足以下条件：h+r≈t其中，h 是头实体的嵌入向量，r 是关系的嵌入向量，而 t 是尾实体的嵌入向量。

训练过程
正样本：对于每个正样本三元组 (h,r,t)，TransE尝试最小化 h+r 和 t 之间的距离。
负样本：为了增强模型的泛化能力，TransE还会考虑负样本，这些负样本通常是通过替换头实体或尾实体得到的。模型需要最大化正样本和负样本之间的距离。
距离度量
通常使用L1或L2范数来度量两个向量之间的距离。例如，使用L1范数的目标函数可以表示为：
    L= (h,r,t)∈S∑(h ′,r ′,t′)∈S′∑[γ+d(h+r,t)−d(h′+r′,t′)] +
​
其中，S是正样本集合，S′是负样本集合，γ是一个边际值，d是距离函数，
[x]+[x]+表示正部分函数（即max(x,0)）。

优点
简单：TransE的模型结构简单，易于理解和实现。
高效：训练过程相对高效，特别是在大规模知识图谱上。
缺点
处理复杂关系的能力有限：TransE在处理一对多、多对一和多对多关系时效果不佳。
无法表示非对称关系：由于使用加法操作，TransE无法有效表示非对称关系。

代码实现流程：
1.训练数据提取：
  将储存在本地文件的txt格式的训练数据进行提取并处理为（h，r，t）的三元组，可通过以下代码进行实现：

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
这样以后就得到了训练数据的三元组，同时计算出了实体数与关系数，实体数为162336，关系数为47，将实体和关系索引存入字典，方便进一步查询

2.定义TransE：

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
在这个代码中，我们实现了TransE的基本框架，包括前向传播及距离

3.损失函数：

    def evaluate(model, triplets, entity_count, batch_size=8192):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i in range(0, len(triplets), batch_size):
            batch_triplets = triplets[i:i + batch_size]
            heads, relations, tails = batch_triplets[:, 0], batch_triplets[:, 1], batch_triplets[:, 2]

            negative_samples = generate_negative_samples(batch_triplets, entity_count, negative_sample_size=1)
            negative_heads, negative_relations, negative_tails = negative_samples[:, 0], negative_samples[:,1], negative_samples[:, 2]

            positive_loss = model.distance(heads, relations, tails).mean()
            negative_loss = model.distance(negative_heads, negative_relations, negative_tails).mean()

            loss = positive_loss + negative_loss
            total_loss += loss.item()
            count += 1
    return total_loss / count

4.训练参数设置：

    ENTITY_NUM = 162336  # 实体的数量
    RELATION_NUM = 47  # 关系的数量
    EMBEDDING_DIM = 5  # 嵌入维度
    LR = 0.1  # 学习率
    BATCH_SIZE = 8192  # 批处理大小

5.开始训练，每训练完成一轮，输出一次训练损失，同时每训练完成十轮进行一次线形图绘制并保存模型权重与优化器设置，便于后续使用及更直观地看到训练损失：
   
    for epoch in range(1000):
    for i in range(0, len(triplets), BATCH_SIZE):
        batch_triplets = triplets[i:i + BATCH_SIZE]
        heads, relations, tails = batch_triplets[:, 0], batch_triplets[:, 1], batch_triplets[:, 2]

        negative_samples = generate_negative_samples(batch_triplets, ENTITY_NUM, negative_sample_size=1)
        negative_heads, negative_relations, negative_tails = negative_samples[:, 0], negative_samples[:,1], negative_samples[:, 2]

        optimizer.zero_grad()

        positive_loss = model.distance(heads, relations, tails).mean()
        negative_loss = model.distance(negative_heads, negative_relations, negative_tails).mean()

        loss = positive_loss + negative_loss
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        train_losses.append(loss.item())
        print(f'Epoch {epoch}: Loss = {loss.item()}')

    if epoch % 10 == 0:
        eval_loss = evaluate(model, triplets, ENTITY_NUM)
        eval_losses.append(eval_loss) 
        print(f'Epoch {epoch}: Evaluation Loss = {eval_loss}')
        plt.figure(figsize=(12, 5))

        model_path = r'D:\pycode\pythonProject2\model\model.pth'  # 模型保存路径
        optimizer_path = r'D:\pycode\pythonProject2\optimizer\optimizer.pth'  # 优化器保存路径

        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(eval_losses, label='Evaluation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Evaluation Loss per Epoch')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

6.引用训练完成后的模型权重model.pth，对json进行处理与重写，修改output中的片段，使其输出最有可能的五个实体或关系（在预测函数中得分最高的五个）：

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

        top_relations = [id_to_relation[idx.item()] for idx in top_indices]  # 使用 id_to_relation 字典
        return top_relations

7.读取json文件获取任务并进行修改:

    for item in data.get('entity_prediction', []):
        head_entity = item['input'][0]
        relation = item['input'][1]
        predicted_entities = predict_entity(model, head_entity, relation, entity_dict, relation_dict,id_to_entity, top_k=5)
        item['output'] = predicted_entities

处理 link_prediction 任务

    for item in data.get('link_prediction', []):
        head_entity = item['input'][0]
        tail_entity = item['input'][1]
        predicted_relations = predict_relation(model, head_entity, tail_entity, entity_dict, relation_dict,id_to_relation,top_k=5)
        item['output'] = predicted_relations
将更新后的数据写回 JSON 文件
 
    with open(r"C:\Users\86159\Desktop\123.json",'w',encoding='utf-8') as file:
        json.dump(data, file,ensure_ascii=False, indent=4)
最后文件保存时采用了file,ensure_ascii=False参数，因为经过测试，用utf-8进行读写后，原文件中的中文字符默认改为了以utf-8编码的形式存在，需要修改回来
