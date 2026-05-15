import h5py
import pickle
import torch
import numpy as np
from layers.layer import *


# Get item2id and write into txt
def write_index_dict(datasets):
    path = 'datasets/'+datasets+'/'
    entities = set()
    relations = set()
    with open(path+datasets+'_EntityTriples.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split(' ')
            entities.add(instance[0])
            relations.add(instance[1])
            entities.add(instance[2])

    with open(path+'entity2id.txt', 'w') as f:
        for index, entity in enumerate(entities):
            f.write(entity+' '+str(index)+'\n')

    with open(path+'relation2id.txt', 'w') as f:
        for index, relation in enumerate(relations):
            f.write(relation+' '+str(index)+'\n')


# Bind img features with id
# def write_img_vec(datasets):
#     path = 'datasets/'+datasets+'/'
#     entities = {}
#     with open(path + datasets + '_ImageIndex.txt', 'r') as f:
#         for line in f:
#             instance = line.strip().split('\t')
#             entities[instance[0]] = instance[1]
#
#     img_features = []
#     with open(path + 'entity2id.txt', 'r') as f:
#         with h5py.File(path + datasets + '_ImageData.h5', 'r') as img:
#             img_all = np.array([feats for feats in img.values()])
#             img_mean = np.mean(img_all.reshape(-1, img_all.shape[2]), 0)
#             for line in f:
#                 instance = line.strip().split(' ')
#                 entity = instance[0]
#                 if entity in entities.keys():
#                     img_features.append(np.array(img[entities[entity]]).flatten())
#                 else:
#                     img_features.append(img_mean)
#
#     img_features = np.array(img_features)
#     pickle.dump(img_features, open(path + 'img_features.pkl', 'wb'))
# 替换 utils/data_util.py 中的 write_img_vec 函数

def write_img_vec(datasets):
    path = 'datasets/' + datasets + '/'
    entities = {}
    with open(path + datasets + '_ImageIndex.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split('\t')
            # 假设此处 instance[1] 是以逗号分隔的多个候选图像索引列表
            entities[instance[0]] = instance[1].split(',') if ',' in instance[1] else [instance[1]]

    img_features = []

    # 模拟外部传入的图结构和特征提取器 (实际应用中需要提前离线计算好)
    # 此处为框架预留的接口映射
    tau_low = 0.6  # 第一阶段预设初始阈值
    gate = GSMSF_Gate(lambda1=0.3, lambda2=0.4, lambda3=0.3, k=5)

    with open(path + 'entity2id.txt', 'r',encoding='utf-8') as f:
        with h5py.File(path + datasets + '_ImageData.h5', 'r') as img:
            img_all = np.array([feats for feats in img.values()])
            global_img_mean = np.mean(img_all.reshape(-1, img_all.shape[2]), 0)

            for line in f:
                instance = line.strip().split(' ')
                entity = instance[0]

                if entity in entities.keys():
                    candidate_indices = entities[entity]
                    candidate_feats = [np.array(img[idx]).flatten() for idx in candidate_indices if idx in img]

                    if len(candidate_feats) > 1:
                        # ---------------------------------------------------------
                        # 第一层级：基于感知哈希的轻量级图像过滤 (公式 4-1)
                        # 采用多数投票法选取参考图像 img_ref (此处简化为首个图像作为基准演示)
                        # 保留 SpHash >= tau_low 的图像
                        # ---------------------------------------------------------
                        filtered_stage1 = candidate_feats  # 伪代码: [feat for feat in candidate_feats if calc_phash(feat, ref) >= tau_low]

                        # ---------------------------------------------------------
                        # 第二层级与第三层级：图结构引导的精筛与自适应阈值
                        # ---------------------------------------------------------
                        # 假设我们获取了候选集节点间的 phash, clip, ssim 以及图结构距离
                        # 这里用随机张量模拟这些离线获取的特征矩阵输入
                        num_candidates = len(filtered_stage1)
                        dummy_phash = torch.rand(num_candidates)
                        dummy_clip = torch.rand(num_candidates)
                        dummy_ssim = torch.rand(num_candidates)
                        dummy_metapath = torch.randint(1, 5, (num_candidates,)).float()
                        dummy_dist = torch.randint(1, 3, (num_candidates,)).float()

                        alpha_ij, tau_e = gate(dummy_phash, dummy_clip, dummy_ssim, dummy_metapath, dummy_dist,
                                               num_candidates)

                        # 依据自适应阈值和综合得分进行筛选
                        valid_mask = (alpha_ij >= tau_e).numpy()
                        final_feats = [feat for idx, feat in enumerate(filtered_stage1) if valid_mask[idx]]

                        # 输出的最终图像集合具有更高的语义一致性，求平均或使用注意力融合
                        if len(final_feats) > 0:
                            img_features.append(np.mean(final_feats, 0))
                        else:
                            img_features.append(np.mean(candidate_feats, 0))  # 退化策略
                    elif len(candidate_feats) == 1:
                        img_features.append(candidate_feats[0])
                    else:
                        img_features.append(global_img_mean)
                else:
                    img_features.append(global_img_mean)

    img_features = np.array(img_features)
    pickle.dump(img_features, open(path + 'img_features.pkl', 'wb'))

def data_preprocess(datasets):
    write_index_dict(datasets)
    write_img_vec(datasets)
    dataset_split(datasets)


def read_entity_from_id(path):
    entity2id = {}
    with open(path + 'entity2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split()
            entity2id[instance[0]] = int(instance[1])

    return entity2id


def read_relation_from_id(path):
    relation2id = {}
    with open(path + 'relation2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split()
            relation2id[instance[0]] = int(instance[1])

    return relation2id


# Calculate adjacency matrix
def get_adj(path, split):
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    triples = []
    rows, cols, data = [], [], []
    unique_entities = set()
    with open(path+split+'.txt', 'r', encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split(' ')
            e1, r, e2 = instance[0], instance[1], instance[2]
            unique_entities.add(e1)
            unique_entities.add(e2)
            triples.append((entity2id[e1], relation2id[r], entity2id[e2]))
            rows.append(entity2id[e2])
            cols.append(entity2id[e1])
            data.append(relation2id[r])

    return triples, (rows, cols, data), unique_entities


# Load data triples and adjacency matrix
def load_data(datasets):
    path = 'datasets/'+datasets+'/'
    train_triples, train_adj, train_unique_entities = get_adj(path, 'train')
    val_triples, val_adj, val_unique_entities = get_adj(path, 'val')
    test_triples, test_adj, test_unique_entities = get_adj(path, 'test')
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    img_features = pickle.load(open(path+'img_features.pkl', 'rb'))
    text_features = pickle.load(open(path+'text_features.pkl', 'rb'))

    return entity2id, relation2id, img_features, text_features, \
           (train_triples, train_adj, train_unique_entities), \
           (val_triples, val_adj, val_unique_entities), \
           (test_triples, test_adj, test_unique_entities)


# Split data into train, val and test
def dataset_split(datasets):
    path = 'datasets/' + datasets + '/'
    with open(path+datasets+'_EntityTriples.txt', 'r',encoding='utf-8') as f:
        triples = f.readlines()

    np.random.shuffle(triples)
    nb_val = round(0.05 * len(triples))
    nb_test = round(0.05 * len(triples))
    val_triples, test_triples, train_triples = triples[:nb_val], triples[nb_val: nb_val+nb_test], triples[nb_val+nb_test:]

    with open(path+'train.txt', 'w') as f:
        f.writelines(train_triples)
    with open(path+'val.txt', 'w') as f:
        f.writelines(val_triples)
    with open(path+'test.txt', 'w') as f:
        f.writelines(test_triples)


def data_loader(datasets):
    path = 'datasets/'+datasets
    with open(path+'/'+datasets+'_EntityTriples.txt', 'r', encoding='utf-8') as f:
        for line in f:
            print(line)



