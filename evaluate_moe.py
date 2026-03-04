import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from utils import *
from Inference_model.Subspace_Inference import *

def calculate_p_at_k(Y_pre, Y_true, k=5):
    """计算P@K"""
    precision_at_k = []
    for i in range(len(Y_pre)):
        top_k_indices = np.argsort(Y_pre[i])[-k:]
        true_in_top_k = np.sum(Y_true[i][top_k_indices])
        precision_at_k.append(true_in_top_k / k)
    return np.mean(precision_at_k)

def evaluate_moe_model():
    """评估MoE候选集模型"""
    print("=== 加载数据和模型 ===")
    
    # 加载数据
    dataClass = torch.load('models/dataClass.pth')
    labDescVec = get_ICD_vectors(dataClass=dataClass, mimicPath="mimic3")
    
    # 生成候选集
    from Inference_model.Global_Inference import DeepLabeler_Contrast
    docEmbedding = dataClass.vector['docEmbedding'] if 'docEmbedding' in dataClass.vector else None
    if docEmbedding is None:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        vector_size = 256
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(dataClass.rawNOTE)]
        model_doc2vec = Doc2Vec(documents, vector_size=vector_size, window=7, min_count=1, workers=4)
        docEmbedding = np.array([model_doc2vec.infer_vector(i) for i in dataClass.rawNOTE], dtype=np.float32)
    
    # 加载Stage1模型生成候选集
    labDescVec_stage1 = -1 + 2 * np.random.random((dataClass.icdNum, 768))
    model_load = DeepLabeler_Contrast(
        dataClass.classNum,
        dataClass.vector['noteEmbedding'],
        docEmbedding,
        labDescVec_stage1,
        cnnHiddenSize=256,
        contextSizeList=[3,4,5],
        docHiddenSize=256,
        dropout=0.75,
        device=torch.device('cuda:0'),
        temp_para=0.05
    )
    model_load.load(path="models/stage1_model.pkl", map_location="cpu", dataClass=dataClass)
    model_load.to_eval_mode()
    
    # 生成候选集
    Candidate_num = 2000
    Y_pre_train_stage1, _ = model_load.calculate_y_prob_by_iterator(
        dataClass.one_epoch_batch_data_stream(8, type='train', device=torch.device("cuda:0"))
    )
    Y_pre_valid_stage1, _ = model_load.calculate_y_prob_by_iterator(
        dataClass.one_epoch_batch_data_stream(8, type='valid', device=torch.device("cuda:0"))
    )
    Y_pre_cnn = np.vstack((Y_pre_train_stage1, Y_pre_valid_stage1))
    sortIdx_t = np.argsort(-Y_pre_cnn, axis=-1)[:, :Candidate_num]
    IdList = np.concatenate((dataClass.trainIdList, dataClass.validIdList))
    candi_t = {}
    for index, item in enumerate(IdList):
        candi_t[item] = sortIdx_t[index]
    
    # 加载MoE模型
    print("=== 加载MoE候选集模型 ===")
    model_eam_s = EAM_S(
        dataClass.classNum, 
        dataClass.vector['noteEmbedding'], 
        labDescVec, dataClass,
        seqMaxLen=4000, 
        chunk_length=400, 
        trans_s=300, 
        attnList=[512],
        embDropout=0.15,
        hdnDropout=0.15, 
        Dropout=0.2, 
        fcDropout=0.3,
        numLayers=2, 
        device=torch.device("cuda:0"),
        use_moe=True,
        num_experts=8,
        num_experts_per_token=2
    )
    
    model_eam_s.load(path="models/stage2_candidate_model_moe.pkl", map_location="cpu", dataClass=dataClass)
    model_eam_s.to_eval_mode()
    
    # 评估函数
    def evaluate_split(split_name):
        print(f"\n=== 评估 {split_name} 集 ===")
        
        Y_pre, Y_true = model_eam_s.calculate_y_prob_by_iterator(
            dataClass.one_epoch_batch_data_stream(16, type=split_name, device=torch.device("cuda:0"), candidate=candi_t)
        )
        
        # 计算各种指标
        results = {}
        
        # 不同阈值下的F1分数
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        for threshold in thresholds:
            Y_pred = (Y_pre > threshold).astype(int)
            f1_micro = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
            precision_micro = precision_score(Y_true, Y_pred, average='micro', zero_division=0)
            recall_micro = recall_score(Y_true, Y_pred, average='micro', zero_division=0)
            
            results[f'threshold_{threshold}'] = {
                'F1-Micro': f1_micro,
                'F1-Macro': f1_macro,
                'Precision-Micro': precision_micro,
                'Recall-Micro': recall_micro
            }
        
        # P@K指标
        for k in [1, 3, 5, 8, 10, 15]:
            p_at_k = calculate_p_at_k(Y_pre, Y_true, k=k)
            results[f'P@{k}'] = p_at_k
        
        # 打印结果
        print(f"\n{split_name} 集评估结果:")
        print("-" * 50)
        for metric, value in results.items():
            if isinstance(value, dict):
                print(f"{metric}:")
                for sub_metric, sub_value in value.items():
                    print(f"  {sub_metric}: {sub_value:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
        
        return results
    
    # 评估各个数据集
    train_results = evaluate_split('train')
    valid_results = evaluate_split('valid')
    test_results = evaluate_split('test')
    
    # 总结
    print("\n=== 评估总结 ===")
    print("=" * 60)
    print("MoE候选集模型性能总结:")
    print(f"候选集大小: {Candidate_num}")
    print(f"专家数量: 8")
    print(f"每个token激活专家数: 2")
    
    # 最佳阈值结果
    best_threshold = 0.3
    print(f"\n最佳阈值 {best_threshold} 下的结果:")
    print(f"验证集 F1-Micro: {valid_results[f'threshold_{best_threshold}']['F1-Micro']:.4f}")
    print(f"测试集 F1-Micro: {test_results[f'threshold_{best_threshold}']['F1-Micro']:.4f}")
    print(f"验证集 P@5: {valid_results['P@5']:.4f}")
    print(f"测试集 P@5: {test_results['P@5']:.4f}")
    
    return {
        'train': train_results,
        'valid': valid_results,
        'test': test_results
    }

if __name__ == "__main__":
    results = evaluate_moe_model() 