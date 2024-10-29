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
