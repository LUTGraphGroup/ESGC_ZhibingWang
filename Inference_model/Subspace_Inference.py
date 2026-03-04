# import torch, time, os, pickle, sys
# from torch import nn
# from torch.nn import functional as F
# import numpy as np
# from collections import *
#
# from transformers import get_cosine_schedule_with_warmup
#
# from nnLayer import *
# from metrics import *
# from torch.nn.init import xavier_uniform_ as xavier_uniform
# from math import floor
# from tqdm import tqdm
# from pytorch_lamb import lamb
# from sklearn.metrics import roc_auc_score
# from Mamba import *
# from graph_models.graph_models import *
# from utils import *
#
#
# class FGM():
#     def __init__(self, model, emb_name='emb'):
#         self.model = model
#         self.emb_name = emb_name
#         self.backup = {}
#
#     def attack(self, epsilon=1.):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and self.emb_name in name:
#                 if param.grad is None:
#                     continue
#                 self.backup[name] = param.data.clone()
#                 norm = torch.norm(param.grad)
#                 if norm != 0:
#                     r_at = epsilon * param.grad / norm
#                     param.data.add_(r_at)
#
#     def restore(self):
#         if not self.backup:
#             return  # 如果没有备份，直接跳过
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and self.emb_name in name:
#                 if name in self.backup:
#                     param.data = self.backup[name]
#         self.backup = {}
#
#
# class EMA():
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
#
#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()
#
#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()
#
#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 self.backup[name] = param.data
#                 param.data = self.shadow[name]
#
#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}
#
#
# class BaseModel:
#     def __init__(self):
#         self.crition = nn.MultiLabelSoftMarginLoss()
#         # 这些属性将在子类中设置
#         self.moduleList = None
#         self.device = None
#
#     def calculate_y_logit(self, X, candidate):
#         pass
#
#     def train(self, dataClass, batchSize, epoch,
#               lr=0.001, momentum=0.9, weightDecay=0.0, stopRounds=10, threshold=0.2, earlyStop=10,
#               savePath='model/KAICD', saveRounds=1, isHigherBetter=True, metrics="MiF", report=["ACC", "MiF"],
#               optimType='Adam', schedulerType='cosine', eta_min=0, warmup_ratio=0.1, dataEnhance=False,
#               dataEnhanceRatio=0.0, attackTrain=False, attackLayerName='emb', ema_para=-1, candidate_para=False):
#         dataClass.dataEnhance = dataEnhance
#         dataClass.dataEnhanceRatio = dataEnhanceRatio
#
#         if attackTrain:
#             self.fgm = FGM(self.moduleList, emb_name=attackLayerName)
#
#         if ema_para > 0:
#             self.ema = EMA(self.moduleList, ema_para)
#         isBeginEMA = False
#
#         metrictor = Metrictor(dataClass.classNum)
#
#         trainStream = dataClass.random_batch_data_stream(batchSize=batchSize, type='train', device=self.device,
#                                                          candidate=candidate_para)
#
#         itersPerEpoch = (dataClass.trainSampleNum + batchSize - 1) // batchSize
#
#         num_training_steps = itersPerEpoch * epoch
#         num_warmup_steps = int(warmup_ratio * itersPerEpoch * epoch)
#
#         optimizer, schedulerRLR = self.get_optimizer(optimType, schedulerType, lr, weightDecay, momentum,
#                                                      num_training_steps, num_warmup_steps, eta_min)
#
#         mtc, bestMtc, stopSteps = 0.0, -1, 0
#
#         if dataClass.validSampleNum > 0:
#             validStream = dataClass.random_batch_data_stream(batchSize=batchSize, type='valid', device=self.device,
#                                                              candidate=candidate_para)
#
#         st = time.time()
#
#         for e in range(epoch):
#             print(f"Epoch {e + 1} with learning rate {optimizer.state_dict()['param_groups'][0]['lr']:.6f}...")
#             print('========== Epoch:%5d ==========' % (e + 1))
#             if (ema_para > 0) and (e > 30) and (not isBeginEMA):  #
#                 self.ema.register()
#                 isBeginEMA = True
#
#             pbar = tqdm(range(itersPerEpoch))
#             for i in pbar:
#                 self.to_train_mode()
#                 X, Y, candidate = next(trainStream)
#                 loss = self._train_step(X, Y, candidate, optimizer, attackTrain, isBeginEMA)
#                 if schedulerRLR != None:
#                     schedulerRLR.step()
#                 pbar.set_description(f"Epoch {e} - Training Loss: {loss.data:.3f}")
#                 if stopRounds > 0 and (e * itersPerEpoch + i + 1) % stopRounds == 0:
#                     self.to_eval_mode()
#                     print("After iters %d: [train] loss= %.3f;" % (e * itersPerEpoch + i + 1, loss), end='')
#                     if dataClass.validSampleNum > 0:
#                         X, Y, candidate = next(validStream)
#                         loss = self.calculate_loss(X, Y, candidate)
#                         print(' [valid] loss= %.3f;' % loss, end='')
#                     restNum = ((itersPerEpoch - i - 1) + (epoch - e - 1) * itersPerEpoch) * batchSize
#                     speed = (e * itersPerEpoch + i + 1) * batchSize / (time.time() - st)
#                     print(" speed: %.3lf items/s; remaining time: %.3lfs;" % (speed, restNum / speed))
#
#             if dataClass.validSampleNum > 0 and (e + 1) % saveRounds == 0:
#                 if isBeginEMA:
#                     self.ema.apply_shadow()
#                 self.to_eval_mode()
#                 print('[Total Valid]', end='')
#                 Y_pre, Y = self.calculate_y_prob_by_iterator(
#                     dataClass.one_epoch_batch_data_stream(batchSize, type='valid', device=self.device,
#                                                           candidate=candidate_para))
#
#                 metrictor.set_data(Y_pre, Y, threshold)
#                 res = metrictor(report)
#                 mtc = res[metrics]
#                 print('=================================')
#                 if (mtc > bestMtc and isHigherBetter) or (mtc < bestMtc and not isHigherBetter):
#                     print('Bingo!!! Get a better Model with val %s: %.3f!!!' % (metrics, mtc))
#                     bestMtc = mtc
#                     self.save("%s.pkl" % savePath, e + 1, bestMtc, dataClass)
#                     stopSteps = 0
#                 else:
#                     stopSteps += 1
#                     if stopSteps >= earlyStop:
#                         print(
#                             'The val %s has not improved for more than %d steps in epoch %d, stop training.' % (metrics,
#                                                                                                                 earlyStop,
#                                                                                                                 e + 1))
#                         break
#             if isBeginEMA:
#                 self.ema.restore()
#         self.load("%s.pkl" % savePath, dataClass=dataClass)
#
#         with torch.no_grad():
#             print(f'============ Result ============')
#             print(f'[Total Train]', end='')
#             Y_pre, Y = self.calculate_y_prob_by_iterator(
#                 dataClass.one_epoch_batch_data_stream(batchSize, type='train', device=self.device,
#                                                       candidate=candidate_para))
#             metrictor.set_data(Y_pre, Y, threshold)
#             metrictor(report)
#             print(f'[Total Valid]', end='')
#             Y_pre, Y = self.calculate_y_prob_by_iterator(
#                 dataClass.one_epoch_batch_data_stream(batchSize, type='valid', device=self.device,
#                                                       candidate=candidate_para))
#             metrictor.set_data(Y_pre, Y, threshold)
#             res = metrictor(report)
#             print(f'================================')
#         return res
#
#     def reset_parameters(self):
#         if self.moduleList is not None:
#             for module in self.moduleList:
#                 for subModule in module.modules():
#                     if hasattr(subModule, "reset_parameters"):
#                         subModule.reset_parameters()
#
#     def save(self, path, epochs, bestMtc=None, dataClass=None):
#         stateDict = {'epochs': epochs, 'bestMtc': bestMtc}
#         for i, module in enumerate(self.moduleList):
#             key = getattr(module, 'name', type(module).__name__ + '_' + str(i))
#             stateDict[key] = module.state_dict()
#         if dataClass is not None:
#             stateDict['trainIdList'], stateDict['validIdList'], stateDict[
#                 'testIdList'] = dataClass.trainIdList, dataClass.validIdList, dataClass.testIdList
#             stateDict['nword2id'], stateDict['tword2id'] = dataClass.nword2id, dataClass.tword2id
#             stateDict['id2nword'], stateDict['id2tword'] = dataClass.id2nword, dataClass.id2tword
#             stateDict['icd2id'], stateDict['id2icd'] = dataClass.icd2id, dataClass.id2icd
#         torch.save(stateDict, path)
#         print(f'Model saved in \"{path}\".')
#
#     def load(self, path, map_location=None, dataClass=None):
#         parameters = torch.load(path, map_location=map_location)
#         for i, module in enumerate(self.moduleList):
#             key = getattr(module, 'name', type(module).__name__ + '_' + str(i))
#             if key in parameters:
#                 module.load_state_dict(parameters[key])
#             else:
#                 print(f"[WARN] State for {key} not found in checkpoint.")
#         if dataClass is not None:
#             dataClass.trainIdList = parameters.get('trainIdList', [])
#             dataClass.validIdList = parameters.get('validIdList', [])
#             dataClass.testIdList = parameters.get('testIdList', [])
#             dataClass.nword2id, dataClass.tword2id = parameters.get('nword2id', {}), parameters.get('tword2id', {})
#             dataClass.id2nword, dataClass.id2tword = parameters.get('id2nword', {}), parameters.get('id2tword', {})
#             dataClass.icd2id, dataClass.id2icd = parameters.get('icd2id', {}), parameters.get('id2icd', {})
#         print(f"{parameters['epochs']} epochs and {parameters['bestMtc']:.3f} val Score's model load finished.")
#
#     def calculate_y_prob(self, X, candidate):
#         out = self.calculate_y_logit(X, candidate)
#         if out is not None and 'y_logit' in out:
#             Y_pre = out['y_logit']
#             return torch.sigmoid(Y_pre)
#         return None
#
#     def calculate_y(self, X, candidate, threshold=0.2):
#         Y_pre = self.calculate_y_prob(X, candidate)
#         if Y_pre is not None:
#             isONE = Y_pre > threshold
#             Y_pre[isONE], Y_pre[~isONE] = 1, 0
#         return Y_pre
#
#     def calculate_loss(self, X, Y, candidate):
#         out = self.calculate_y_logit(X, candidate)
#         if out is not None and 'y_logit' in out:
#             Y_logit = out['y_logit']
#             addLoss = 0.0
#             if 'loss' in out: addLoss += out['loss']
#             return self.crition(Y_logit, Y) + addLoss
#         return None
#
#     def calculate_indicator_by_iterator(self, dataStream, classNum, report, threshold):
#         metrictor = Metrictor(classNum)
#         Y_prob_pre, Y = self.calculate_y_prob_by_iterator(dataStream)
#         metrictor.set_data(Y_prob_pre, Y, threshold)
#         return metrictor(report)
#
#     def calculate_y_prob_by_iterator(self, dataStream):
#         YArr, Y_preArr = [], []
#         while True:
#             try:
#                 X, Y, candidate = next(dataStream)
#             except:
#                 break
#             Y_pre = self.calculate_y_prob(X, candidate).cpu().data.numpy().astype(np.float16)
#             Y = Y.cpu().data.numpy().astype(np.int32)
#             YArr.append(Y)
#             Y_preArr.append(Y_pre)
#         YArr, Y_preArr = np.vstack(YArr), np.vstack(Y_preArr)
#         return Y_preArr, YArr
#
#     def calculate_y_by_iterator(self, dataStream, threshold=0.2):
#         Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
#         isONE = Y_preArr > threshold
#         Y_preArr[isONE], Y_preArr[~isONE] = 1, 0
#         return Y_preArr, YArr
#
#     def to_train_mode(self):
#         for module in self.moduleList:
#             module.train()
#
#     def to_eval_mode(self):
#         for module in self.moduleList:
#             module.eval()
#
#     def _train_step(self, X, Y, candidate, optimizer, attackTrain, isBeginEMA):
#         loss = self.calculate_loss(X, Y, candidate)
#         loss.backward()
#         if attackTrain:
#             self.fgm.attack()  # 在embedding上添加对抗梯度
#             lossAdv = self.calculate_loss(X, Y, candidate)
#             lossAdv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#             self.fgm.restore()  # 恢复embedding参数
#         nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
#         optimizer.step()
#         if isBeginEMA:
#             self.ema.update()
#         optimizer.zero_grad()
#         return loss
#
#     def get_optimizer(self, optimType, schedulerType, lr, weightDecay, momentum, num_training_steps, num_warmup_steps,
#                       eta_min):
#
#         # Prepare optimizer and schedule (linear warmup and decay)
#         model_lr = {'others': lr}
#         no_decay = ['bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = []
#         for layer_name in model_lr:
#             lr = model_lr[layer_name]
#             if layer_name != 'others':
#                 optimizer_grouped_parameters += [
#                     {
#                         "params": [p for n, p in self.moduleList.named_parameters() if
#                                    (not any(nd in n for nd in no_decay)
#                                     and layer_name in n)],
#                         "weight_decay": weightDecay,
#                         "lr": lr,
#                     },
#                     {
#                         "params": [p for n, p in self.moduleList.named_parameters() if (any(nd in n for nd in no_decay)
#                                                                                         and layer_name in n)],
#                         "weight_decay": 0.0,
#                         "lr": lr,
#                     },
#                 ]
#             else:
#                 optimizer_grouped_parameters += [
#                     {
#                         "params": [p for n, p in self.moduleList.named_parameters() if
#                                    (not any(nd in n for nd in no_decay)
#                                     and not any(name in n for name in model_lr))],
#                         "weight_decay": weightDecay,
#                         "lr": lr,
#                     },
#                     {
#                         "params": [p for n, p in self.moduleList.named_parameters() if (any(nd in n for nd in no_decay)
#                                                                                         and not any(
#                                     name in n for name in model_lr))],
#                         "weight_decay": 0.0,
#                         "lr": lr,
#                     },
#                 ]
#
#         if optimType == 'Adam':
#             optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)
#         elif optimType == 'AdamW':
#             optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)
#         elif optimType == 'Lamb':
#             optimizer = lamb.Lamb(optimizer_grouped_parameters, weight_decay=weightDecay)
#         elif optimType == 'SGD':
#             optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=momentum,
#                                         weight_decay=weightDecay)
#         elif optimType == 'Adadelta':
#             optimizer = torch.optim.Adadelta(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)
#
#         print("len(optimizer_grouped_parameters))")
#         print(len(optimizer_grouped_parameters))
#
#         if schedulerType == 'cosine':
#             schedulerRLR = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps,
#                                                            num_warmup_steps=num_warmup_steps)
#         elif schedulerType == 'cosine_Anneal':
#             schedulerRLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, T_mult=3, eta_min=eta_min)
#         elif schedulerType == 'None':
#             schedulerRLR = None
#
#         return optimizer, schedulerRLR
#
#
#
# class FLASH_ICD_Candidates_2Inputs(BaseModel):
#     def __init__(self, classNum, embedding, labDescVec, dataClass,
#                  seqMaxLen, chunk_length, trans_s, attnList=[],
#                  embDropout=0.2, hdnDropout=0.2, Dropout=0.3, fcDropout=0.5, numLayers=1,
#                  device=torch.device('cuda:0'), useCircleLoss=False, compress=False,
#                  graph_path='mimic3/full_relation.pkl', useFocalLoss=False, fusionDropout=0.2):
#         super().__init__()
#         self.useFocalLoss = useFocalLoss
#         self.fusion_dropout = nn.Dropout(fusionDropout)
#         self.emb_dim = embedding.shape[1]
#         self.lab_dim = labDescVec.shape[1]
#         self.global_class_num = 1000  # 固定为1000类
#         self.kept_global_indices = torch.arange(1000, device=device)  # 保持前1000个索引
#         self.device = device
#         # === 改进的文本编码器 ===
#         self.embedding = TextEmbedding_1d(embedding, dropout=embDropout).to(device)
#         self.mamba_encoder = MambaStack(
#             d_model=self.emb_dim,
#             d_state=16,  # 增加状态维度
#             d_conv=4,  # 增加卷积维度
#             expand=2,  # 增加扩展因子
#             num_layers=10,  # 增加层数
#             dropout=0.2  # 降低dropout
#         ).to(device)
#         self.mamba_encoder.name = 'mamba_encoder'
#
#         # === 添加层归一化 ===
#         self.layer_norm = nn.LayerNorm(self.emb_dim).to(device)
#
#         # === 改进的标签编码器 ===
#         self.labDescVec = torch.tensor(labDescVec, dtype=torch.float32).to(device)
#         self.icdAttn = DeepICDDescCandiAttention(
#             inSize=self.emb_dim,
#             classNum=classNum,
#             labSize=self.lab_dim,
#             hdnDropout=hdnDropout,
#             attnList=attnList,
#             labDescVec=labDescVec
#         ).to(device)
#
#         # === 改进的分类头 ===
#         self.fcLinear = MLP(labDescVec.shape[1], 1, [512, 256], dropout=0.3).to(device)
#
#         # === 简化图编码器 ===
#         with open(graph_path, "rb") as f:
#             edges_dict = pickle.load(f)
#
#         # 从图中提取所有代码
#         graph_nodes = set()
#         for (code1, code2) in edges_dict.keys():
#             graph_nodes.add(code1)
#             graph_nodes.add(code2)
#
#         all_codes = sorted(list(graph_nodes))
#         c2ind = {code: idx for idx, code in enumerate(all_codes)}
#
#         # 建立从dataClass索引到图中索引的映射
#         self.dataclass_to_graph_mapping = {}
#         for i, code in enumerate(dataClass.id2icd):
#             if code in c2ind:
#                 self.dataclass_to_graph_mapping[i] = c2ind[code]
#             else:
#                 self.dataclass_to_graph_mapping[i] = 0
#
#         # 构建 cm2ind：major_code -> 索引
#         def extract_major_code(code):
#             code_str = str(code)
#             parts = code_str.split('_')
#             if len(parts) >= 2:
#                 if len(parts[1]) <= 3:
#                     return code_str
#                 else:
#                     return f"{parts[0]}_{parts[1][:3]}"
#             return code_str
#
#         major_codes = set()
#         for code in all_codes:
#             major_code = extract_major_code(code)
#             major_codes.add(major_code)
#
#         major_codes = sorted(list(major_codes))
#         cm2ind = {code: idx for idx, code in enumerate(major_codes)}
#
#         print(f"[INFO] dataClass.id2icd size = {len(dataClass.id2icd)}")
#         print(f"[INFO] 图中代码总数 = {len(all_codes)}")
#         print(f"[INFO] c2ind size = {len(c2ind)}")
#         print(f"[INFO] cm2ind size = {len(cm2ind)}")
#
#         self.graph_encoder = KGCodeReassign(
#             args={
#                 'attention_dim': 768,
#                 'use_multihead': 4,
#                 'edge_dim': 16,
#                 'rep_dropout': Dropout,
#             },
#             edges_dict=edges_dict,
#             c2ind=c2ind,
#             cm2ind=cm2ind
#         ).to(device)
#
#         # === 简化图融合机制 ===
#         self.w_graph_linear = MLP(
#             inSize=768,
#             outSize=768,
#             hiddenList=[768],
#             dropout=0.2,
#             actFunc=nn.ReLU
#         ).to(device)
#
#         # === 简化的门控融合 ===
#         # 输入维度是 lab_dim + 768 = 拼接后的维度
#         gate_input_dim = self.lab_dim + 768
#         self.gate_fusion = nn.Sequential(
#             nn.Linear(gate_input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         ).to(device)
#
#         # === Module list ===
#         self.moduleList = nn.ModuleList([
#             self.embedding,
#             self.mamba_encoder,
#             self.layer_norm,
#             self.icdAttn,
#             self.fcLinear,
#             self.graph_encoder,
#             self.w_graph_linear,
#             self.gate_fusion,
#             self.fusion_dropout
#         ])
#
#         # 改进的损失函数
#
#         if useCircleLoss:
#             self.crition = MultiLabelCircleLoss()
#         elif hasattr(self, 'useFocalLoss') and self.useFocalLoss:
#             self.crition = MultiLabelFocalLoss()  # 使用多标签Focal损失
#         else:
#             self.crition = nn.MultiLabelSoftMarginLoss()
#
#         self.hdnDropout = hdnDropout
#         self.fcDropout = fcDropout
#         self.classNum = classNum
#
#     def calculate_y_logit(self, X, candidate):
#         candidate = torch.tensor(candidate, dtype=torch.long, device=self.device) if not torch.is_tensor(
#             candidate) else candidate.long()
#         x = self.embedding(X['noteArr'])  # [B, L, D]
#         x = self.mamba_encoder(x)  # [B, L, D]
#         x = self.layer_norm(x)  # [B, L, D]
#         x = self.icdAttn(x, candidate=candidate, labDescVec=self.labDescVec)  # [B, C, D]
#         x = self.fcLinear(x).squeeze(2)  # [B, C]
#         return {'y_logit': x}
#
# class FLASH_ICD_Candidates_2Inputs(BaseModel):
#     def __init__(self, classNum, embedding, labDescVec, dataClass,
#                  seqMaxLen, chunk_length, trans_s, attnList=[],
#                  embDropout=0.2, hdnDropout=0.2, Dropout=0.3, fcDropout=0.5, numLayers=1,
#                  device=torch.device('cuda:0'), useCircleLoss=False, compress=False,
#                  graph_path='mimic3/full_relation.pkl', useFocalLoss=True, fusionDropout=0.2):
#         super().__init__()
#         self.useFocalLoss = useFocalLoss
#         self.fusion_dropout = nn.Dropout(fusionDropout)
#         self.emb_dim = embedding.shape[1]
#         self.lab_dim = labDescVec.shape[1]
#         self.device = device
#
#         # === 改进的文本编码器 ===
#         self.embedding = TextEmbedding_1d(embedding, dropout=embDropout).to(device)
#         self.mamba_encoder = MambaStack1(
#             d_model=self.emb_dim,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#             num_layers=12,  # 可调层数
#             dropout=0.2,  # 可调Dropout
#             use_moe_enhancement=True,  # 启用MoE增强
#             moe_weight=0.01  # 很小的权重，保持原始功能为主
#         ).to(device)
#         self.mamba_encoder.name = 'mamba_encoder'
#
#         # === 改进的标签编码器 ==
#         self.labDescVec = torch.tensor(labDescVec, dtype=torch.float32).to(device)
#         self.icdAttn = DeepICDDescCandiAttention(
#             inSize=self.emb_dim,
#             classNum=classNum,
#             labSize=self.lab_dim,
#             hdnDropout=hdnDropout,
#             attnList=attnList,
#             labDescVec=labDescVec
#         ).to(device)
#         # === 改进的分类头 ===
#         self.fcLinear = MLP(labDescVec.shape[1], 1, [512, 256], dropout=0.3).to(device)
#         # === 简化图编码器 ===
#         with open(graph_path, "rb") as f:
#             edges_dict = pickle.load(f)
#         # 从图中提取所有代码
#         graph_nodes = set()
#         for (code1, code2) in edges_dict.keys():
#             graph_nodes.add(code1)
#             graph_nodes.add(code2)
#         all_codes = sorted(list(graph_nodes))
#         c2ind = {code: idx for idx, code in enumerate(all_codes)}
#         # 建立从dataClass索引到图中索引的映射
#         self.dataclass_to_graph_mapping = {}
#         for i, code in enumerate(dataClass.id2icd):
#             if code in c2ind:
#                 self.dataclass_to_graph_mapping[i] = c2ind[code]
#             else:
#                 self.dataclass_to_graph_mapping[i] = 0
#         # 构建 cm2ind：major_code -> 索引
#         def extract_major_code(code):
#             code_str = str(code)
#             parts = code_str.split('_')
#             if len(parts) >= 2:
#                 if len(parts[1]) <= 3:
#                     return code_str
#                 else:
#                     return f"{parts[0]}_{parts[1][:3]}"
#             return code_str
#         major_codes = set()
#         for code in all_codes:
#             major_code = extract_major_code(code)
#             major_codes.add(major_code)
#         major_codes = sorted(list(major_codes))
#         cm2ind = {code: idx for idx, code in enumerate(major_codes)}
#         print(f"[INFO] dataClass.id2icd size = {len(dataClass.id2icd)}")
#         print(f"[INFO] 图中代码总数 = {len(all_codes)}")
#         print(f"[INFO] c2ind size = {len(c2ind)}")
#         print(f"[INFO] cm2ind size = {len(cm2ind)}")
#         self.graph_encoder = KGCodeReassign(
#             args={
#                 'attention_dim': 768,
#                 'use_multihead': 4,
#                 'edge_dim': 16,
#                 'rep_dropout': Dropout,
#             },
#             edges_dict=edges_dict,
#             c2ind=c2ind,
#             cm2ind=cm2ind
#         ).to(device)
#         # === 简化图融合机制 ===
#         self.w_graph_linear = MLP(
#             inSize=768,
#             outSize=768,
#             hiddenList=[768],
#             dropout=0.2,
#             actFunc=nn.ReLU
#         ).to(device)
#         # === 简化的门控融合 ===
#         self.gate_fusion = nn.Sequential(
#             nn.Linear(1536, 256),  # 2 * 768 = 1536
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         ).to(device)
#         # === Module list ===
#         self.moduleList = nn.ModuleList([
#             self.embedding,
#             self.mamba_encoder,
#             self.icdAttn,
#             self.fcLinear,
#             self.graph_encoder,
#             self.w_graph_linear,
#             self.gate_fusion,
#             self.fusion_dropout
#         ])
#
#         # 改进的损失函数
#
#         if useCircleLoss:
#             self.crition = MultiLabelCircleLoss()
#         elif hasattr(self, 'useFocalLoss') and self.useFocalLoss:
#             self.crition = MultiLabelFocalLoss()
#         else:
#             self.crition = nn.MultiLabelSoftMarginLoss()
#
#         self.hdnDropout = hdnDropout
#         self.fcDropout = fcDropout
#         self.classNum = classNum
#
#     def calculate_y_logit(self, X, candidate):
#         candidate = torch.tensor(candidate, dtype=torch.long, device=self.device) if not torch.is_tensor(
#             candidate) else candidate.long()
#         x = self.embedding(X['noteArr'])  # [B, L, D]
#         x = self.mamba_encoder(x)  # [B, L, D]
#         x = self.icdAttn(x, candidate=candidate, labDescVec=self.labDescVec)  # [B, C, D]
#         x = self.fcLinear(x).squeeze(2)  # [B, C]
#         return {'y_logit': x}
#
#     def calculate_y_prob(self, X, candidate):
#         # === 简化的candidate处理 ===
#         if not torch.is_tensor(candidate):
#             candidate = torch.as_tensor(candidate, device=self.device)
#         candidate = candidate.long()
#
#         # === 改进的文本特征提取 ===
#         note_repr = self.embedding(X['noteArr'])  # [B, L, D]
#         note_repr = self.mamba_encoder(note_repr)  # [B, L, D]
#
#         m = self.icdAttn(note_repr, candidate=candidate, labDescVec=self.labDescVec)  # [B, C, D]
#
#         # === 原始预测 ===
#         y_logit_orig = self.calculate_y_logit(X, candidate)['y_logit']  # [B, C]
#         p_hat = torch.sigmoid(y_logit_orig)  # [B, C]
#
#         # === 简化的图增强 ===
#         if candidate.dim() == 2:
#             graph_indices = torch.tensor(
#                 [[self.dataclass_to_graph_mapping[idx.item()] for idx in row] for row in candidate],
#                 device=self.device
#             )
#         elif candidate.dim() == 3:
#             graph_indices = torch.tensor(
#                 [[[self.dataclass_to_graph_mapping[idx.item()] for idx in row] for row in batch] for batch in
#                  candidate],
#                 device=self.device
#             )
#         else:
#             graph_indices = candidate
#
#         m_updated = self.graph_encoder(m, m, indices=graph_indices)  # [B, C, D]
#         w_graph = self.w_graph_linear(m_updated)  # [B, C, D]
#         logits_graph = torch.einsum("bcd,bcd->bc", m_updated, w_graph)
#         p_graph = torch.sigmoid(logits_graph)  # [B, C]
#
#         # === 简化的门控融合 ===
#         gate_input = self.fusion_dropout(torch.cat([m, m_updated], dim=-1))  # [B, C, 2D]
#         B, C, D2 = gate_input.shape
#         gate_input_reshaped = gate_input.view(-1, D2)  # [B*C, 2D]
#         gate = self.gate_fusion(gate_input_reshaped)  # [B*C, 1]
#         gate = gate.view(B, C)  # [B, C]
#         p_fused = (1 - gate) * p_hat + gate * p_graph
#
#         # === 填充完整预测矩阵 ===
#         # B = p_hat.size(0)
#         # Zero_matrix = torch.zeros(B, self.classNum, device=self.device)
#         # Y_pre = Zero_matrix.scatter_(dim=1, index=candidate, src=p_hat)
#         # return Y_pre
#         B = p_fused.size(0)
#         Zero_matrix = torch.zeros(B, self.classNum, device=self.device)
#         Y_pre = Zero_matrix.scatter_(dim=1, index=candidate, src=p_fused)
#         return Y_pre
#
import torch, time, os, pickle, sys
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import *

from transformers import get_cosine_schedule_with_warmup

from nnLayer import *
from metrics import *
from torch.nn.init import xavier_uniform_ as xavier_uniform
from math import floor
from tqdm import tqdm
from pytorch_lamb import lamb
from sklearn.metrics import roc_auc_score
from Mamba import *
from graph_models.graph_models import *
from utils import *


class FGM():
    def __init__(self, model, emb_name='emb'):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}

    def attack(self, epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if param.grad is None:
                    continue
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        if not self.backup:
            return  # 如果没有备份，直接跳过
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class BaseModel:
    def __init__(self):
        self.crition = nn.MultiLabelSoftMarginLoss()
        # 这些属性将在子类中设置
        self.moduleList = None
        self.device = None

    def calculate_y_logit(self, X, candidate):
        pass

    def train(self, dataClass, batchSize, epoch,
              lr=0.001, momentum=0.9, weightDecay=0.0, stopRounds=10, threshold=0.2, earlyStop=10,
              savePath='model/KAICD', saveRounds=1, isHigherBetter=True, metrics="MiF", report=["ACC", "MiF"],
              optimType='Adam', schedulerType='cosine', eta_min=0, warmup_ratio=0.1, dataEnhance=False,
              dataEnhanceRatio=0.0, attackTrain=False, attackLayerName='emb', ema_para=-1, candidate_para=False):
        dataClass.dataEnhance = dataEnhance
        dataClass.dataEnhanceRatio = dataEnhanceRatio

        if attackTrain:
            self.fgm = FGM(self.moduleList, emb_name=attackLayerName)

        if ema_para > 0:
            self.ema = EMA(self.moduleList, ema_para)
        isBeginEMA = False

        metrictor = Metrictor(dataClass.classNum)

        trainStream = dataClass.random_batch_data_stream(batchSize=batchSize, type='train', device=self.device,
                                                         candidate=candidate_para)

        itersPerEpoch = (dataClass.trainSampleNum + batchSize - 1) // batchSize

        num_training_steps = itersPerEpoch * epoch
        num_warmup_steps = int(warmup_ratio * itersPerEpoch * epoch)

        optimizer, schedulerRLR = self.get_optimizer(optimType, schedulerType, lr, weightDecay, momentum,
                                                     num_training_steps, num_warmup_steps, eta_min)

        mtc, bestMtc, stopSteps = 0.0, -1, 0

        if dataClass.validSampleNum > 0:
            validStream = dataClass.random_batch_data_stream(batchSize=batchSize, type='valid', device=self.device,
                                                             candidate=candidate_para)

        st = time.time()

        for e in range(epoch):
            print(f"Epoch {e + 1} with learning rate {optimizer.state_dict()['param_groups'][0]['lr']:.6f}...")
            print('========== Epoch:%5d ==========' % (e + 1))
            if (ema_para > 0) and (e > 30) and (not isBeginEMA):  #
                self.ema.register()
                isBeginEMA = True

            pbar = tqdm(range(itersPerEpoch))
            for i in pbar:
                self.to_train_mode()
                X, Y, candidate = next(trainStream)
                loss = self._train_step(X, Y, candidate, optimizer, attackTrain, isBeginEMA)
                if schedulerRLR != None:
                    schedulerRLR.step()
                pbar.set_description(f"Epoch {e} - Training Loss: {loss.data:.3f}")
                if stopRounds > 0 and (e * itersPerEpoch + i + 1) % stopRounds == 0:
                    self.to_eval_mode()
                    print("After iters %d: [train] loss= %.3f;" % (e * itersPerEpoch + i + 1, loss), end='')
                    if dataClass.validSampleNum > 0:
                        X, Y, candidate = next(validStream)
                        loss = self.calculate_loss(X, Y, candidate)
                        print(' [valid] loss= %.3f;' % loss, end='')
                    restNum = ((itersPerEpoch - i - 1) + (epoch - e - 1) * itersPerEpoch) * batchSize
                    speed = (e * itersPerEpoch + i + 1) * batchSize / (time.time() - st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;" % (speed, restNum / speed))

            if dataClass.validSampleNum > 0 and (e + 1) % saveRounds == 0:
                if isBeginEMA:
                    self.ema.apply_shadow()
                self.to_eval_mode()
                print('[Total Valid]', end='')
                Y_pre, Y = self.calculate_y_prob_by_iterator(
                    dataClass.one_epoch_batch_data_stream(batchSize, type='valid', device=self.device,
                                                          candidate=candidate_para))

                metrictor.set_data(Y_pre, Y, threshold)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc > bestMtc and isHigherBetter) or (mtc < bestMtc and not isHigherBetter):
                    print('Bingo!!! Get a better Model with val %s: %.3f!!!' % (metrics, mtc))
                    bestMtc = mtc
                    self.save("%s.pkl" % savePath, e + 1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps >= earlyStop:
                        print(
                            'The val %s has not improved for more than %d steps in epoch %d, stop training.' % (metrics,
                                                                                                                earlyStop,
                                                                                                                e + 1))
                        break
            if isBeginEMA:
                self.ema.restore()
        self.load("%s.pkl" % savePath, dataClass=dataClass)

        with torch.no_grad():
            print(f'============ Result ============')
            print(f'[Total Train]', end='')
            Y_pre, Y = self.calculate_y_prob_by_iterator(
                dataClass.one_epoch_batch_data_stream(batchSize, type='train', device=self.device,
                                                      candidate=candidate_para))
            metrictor.set_data(Y_pre, Y, threshold)
            metrictor(report)
            print(f'[Total Valid]', end='')
            Y_pre, Y = self.calculate_y_prob_by_iterator(
                dataClass.one_epoch_batch_data_stream(batchSize, type='valid', device=self.device,
                                                      candidate=candidate_para))
            metrictor.set_data(Y_pre, Y, threshold)
            res = metrictor(report)
            print(f'================================')
        return res

    def reset_parameters(self):
        if self.moduleList is not None:
            for module in self.moduleList:
                for subModule in module.modules():
                    if hasattr(subModule, "reset_parameters"):
                        subModule.reset_parameters()

    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc}
        for i, module in enumerate(self.moduleList):
            key = getattr(module, 'name', type(module).__name__ + '_' + str(i))
            stateDict[key] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'], stateDict['validIdList'], stateDict[
                'testIdList'] = dataClass.trainIdList, dataClass.validIdList, dataClass.testIdList
            stateDict['nword2id'], stateDict['tword2id'] = dataClass.nword2id, dataClass.tword2id
            stateDict['id2nword'], stateDict['id2tword'] = dataClass.id2nword, dataClass.id2tword
            stateDict['icd2id'], stateDict['id2icd'] = dataClass.icd2id, dataClass.id2icd
        torch.save(stateDict, path)
        print(f'Model saved in \"{path}\".')

    # def load(self, path, map_location=None, dataClass=None):
    #     parameters = torch.load(path, map_location=map_location)
    #     for i, module in enumerate(self.moduleList):
    #         key = getattr(module, 'name', type(module).__name__ + '_' + str(i))
    #         if key in parameters:
    #             module.load_state_dict(parameters[key])
    #         else:
    #             print(f"[WARN] State for {key} not found in checkpoint.")
    #     if dataClass is not None:
    #         dataClass.trainIdList = parameters.get('trainIdList', [])
    #         dataClass.validIdList = parameters.get('validIdList', [])
    #         dataClass.testIdList = parameters.get('testIdList', [])
    #         dataClass.nword2id, dataClass.tword2id = parameters.get('nword2id', {}), parameters.get('tword2id', {})
    #         dataClass.id2nword, dataClass.id2tword = parameters.get('id2nword', {}), parameters.get('id2tword', {})
    #         dataClass.icd2id, dataClass.id2icd = parameters.get('icd2id', {}), parameters.get('id2icd', {})
    #     print(f"{parameters['epochs']} epochs and {parameters['bestMtc']:.3f} val Score's model load finished.")
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for i, module in enumerate(self.moduleList):
            key = getattr(module, 'name', type(module).__name__ + '_' + str(i))
            if key in parameters:
                try:
                    # 先尝试严格加载
                    module.load_state_dict(parameters[key], strict=True)
                except RuntimeError as e:
                    # 如果严格加载失败（结构不匹配），使用非严格加载
                    print(f"[WARN] 严格加载 {key} 失败，使用非严格加载（跳过不匹配的层）")
                    print(f"[WARN] 错误详情: {str(e)[:200]}...")
                    missing_keys, unexpected_keys = module.load_state_dict(
                        parameters[key], strict=False
                    )
                    if missing_keys:
                        print(f"[WARN] {key} 缺少以下键（将使用随机初始化）: {len(missing_keys)} 个")
                        if len(missing_keys) <= 10:
                            print(f"      详细列表: {missing_keys}")
                    if unexpected_keys:
                        print(f"[WARN] {key} 包含以下未使用的键: {len(unexpected_keys)} 个")
                        if len(unexpected_keys) <= 10:
                            print(f"      详细列表: {unexpected_keys}")
            else:
                print(f"[WARN] State for {key} not found in checkpoint.")
        if dataClass is not None:
            dataClass.trainIdList = parameters.get('trainIdList', [])
            dataClass.validIdList = parameters.get('validIdList', [])
            dataClass.testIdList = parameters.get('testIdList', [])
            dataClass.nword2id, dataClass.tword2id = parameters.get('nword2id', {}), parameters.get('tword2id', {})
            dataClass.id2nword, dataClass.id2tword = parameters.get('id2nword', {}), parameters.get('id2tword', {})
            dataClass.icd2id, dataClass.id2icd = parameters.get('icd2id', {}), parameters.get('id2icd', {})
        print(f"{parameters['epochs']} epochs and {parameters['bestMtc']:.3f} val Score's model load finished.")

    def calculate_y_prob(self, X, candidate):
        out = self.calculate_y_logit(X, candidate)
        if out is not None and 'y_logit' in out:
            Y_pre = out['y_logit']
            return torch.sigmoid(Y_pre)
        return None

    def calculate_y(self, X, candidate, threshold=0.2):
        Y_pre = self.calculate_y_prob(X, candidate)
        if Y_pre is not None:
            isONE = Y_pre > threshold
            Y_pre[isONE], Y_pre[~isONE] = 1, 0
        return Y_pre

    def calculate_loss(self, X, Y, candidate):
        out = self.calculate_y_logit(X, candidate)
        if out is not None and 'y_logit' in out:
            Y_logit = out['y_logit']
            addLoss = 0.0
            if 'loss' in out: addLoss += out['loss']
            return self.crition(Y_logit, Y) + addLoss
        return None

    def calculate_indicator_by_iterator(self, dataStream, classNum, report, threshold):
        metrictor = Metrictor(classNum)
        Y_prob_pre, Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y, threshold)
        return metrictor(report)

    def calculate_y_prob_by_iterator(self, dataStream):
        YArr, Y_preArr = [], []
        while True:
            try:
                X, Y, candidate = next(dataStream)
            except:
                break
            Y_pre = self.calculate_y_prob(X, candidate).cpu().data.numpy().astype(np.float16)
            Y = Y.cpu().data.numpy().astype(np.int32)
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr, Y_preArr = np.vstack(YArr), np.vstack(Y_preArr)
        return Y_preArr, YArr

    def calculate_y_by_iterator(self, dataStream, threshold=0.2):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        isONE = Y_preArr > threshold
        Y_preArr[isONE], Y_preArr[~isONE] = 1, 0
        return Y_preArr, YArr

    def to_train_mode(self):
        for module in self.moduleList:
            module.train()

    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    #
    # def _train_step(self, X, Y, candidate, optimizer, attackTrain, isBeginEMA):
    #     loss = self.calculate_loss(X, Y, candidate)
    #     loss.backward()
    #     if attackTrain:
    #         self.fgm.attack()  # 在embedding上添加对抗梯度
    #         lossAdv = self.calculate_loss(X, Y, candidate)
    #         lossAdv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    #         self.fgm.restore()  # 恢复embedding参数
    #     nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
    #     optimizer.step()
    #     if isBeginEMA:
    #         self.ema.update()
    #     optimizer.zero_grad()
    #     return loss
    def _train_step(self, X, Y, candidate, optimizer, attackTrain, isBeginEMA):
        loss = self.calculate_loss(X, Y, candidate)
        loss.backward()

        # 更合理的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=1.0, norm_type=2)

        if attackTrain:
            self.fgm.attack()
            lossAdv = self.calculate_loss(X, Y, candidate)
            if lossAdv is not None and not torch.isnan(lossAdv):
                lossAdv.backward()
            self.fgm.restore()

        optimizer.step()
        if isBeginEMA:
            self.ema.update()
        optimizer.zero_grad()
        return loss

    def get_optimizer(self, optimType, schedulerType, lr, weightDecay, momentum, num_training_steps, num_warmup_steps,
                      eta_min):

        # Prepare optimizer and schedule (linear warmup and decay)
        model_lr = {'others': lr}
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        for layer_name in model_lr:
            lr = model_lr[layer_name]
            if layer_name != 'others':
                optimizer_grouped_parameters += [
                    {
                        "params": [p for n, p in self.moduleList.named_parameters() if
                                   (not any(nd in n for nd in no_decay)
                                    and layer_name in n)],
                        "weight_decay": weightDecay,
                        "lr": lr,
                    },
                    {
                        "params": [p for n, p in self.moduleList.named_parameters() if (any(nd in n for nd in no_decay)
                                                                                        and layer_name in n)],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]
            else:
                optimizer_grouped_parameters += [
                    {
                        "params": [p for n, p in self.moduleList.named_parameters() if
                                   (not any(nd in n for nd in no_decay)
                                    and not any(name in n for name in model_lr))],
                        "weight_decay": weightDecay,
                        "lr": lr,
                    },
                    {
                        "params": [p for n, p in self.moduleList.named_parameters() if (any(nd in n for nd in no_decay)
                                                                                        and not any(
                                    name in n for name in model_lr))],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]

        if optimType == 'Adam':
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)
        elif optimType == 'AdamW':
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)
        elif optimType == 'Lamb':
            optimizer = lamb.Lamb(optimizer_grouped_parameters, weight_decay=weightDecay)
        elif optimType == 'SGD':
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=momentum,
                                        weight_decay=weightDecay)
        elif optimType == 'Adadelta':
            optimizer = torch.optim.Adadelta(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)

        print("len(optimizer_grouped_parameters))")
        print(len(optimizer_grouped_parameters))

        if schedulerType == 'cosine':
            schedulerRLR = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps,
                                                           num_warmup_steps=num_warmup_steps)
        elif schedulerType == 'cosine_Anneal':
            schedulerRLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, T_mult=3, eta_min=eta_min)
        elif schedulerType == 'None':
            schedulerRLR = None

        return optimizer, schedulerRLR


class EAM_S(BaseModel):
    def __init__(self, classNum, embedding, labDescVec, dataClass,
                 seqMaxLen, chunk_length, trans_s, attnList=[],
                 embDropout=0.2, hdnDropout=0.2, Dropout=0.3, fcDropout=0.5, numLayers=1,
                 device=torch.device('cuda:0'), useCircleLoss=False, compress=False,
                 graph_path='mimic3/full_relation.pkl', useFocalLoss=False, fusionDropout=0.2,
                 stage1_model=None):
        super().__init__()
        self.useFocalLoss = useFocalLoss
        self.fusion_dropout = nn.Dropout(fusionDropout)
        self.emb_dim = embedding.shape[1]
        self.lab_dim = labDescVec.shape[1]
        self.device = device
        self.stage1_model = stage1_model  # 保存stage1模型引用

        # === 改进的文本编码器 ===
        self.embedding = TextEmbedding_1d(embedding, dropout=embDropout).to(device)
        self.mamba_encoder = MambaStack1(
            d_model=self.emb_dim,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=12,  # 可调层数
            dropout=0.2,  # 可调Dropout
            use_moe_enhancement=True,  # 启用MoE增强
            moe_weight=0.01
        ).to(device)
        self.mamba_encoder.name = 'mamba_encoder'
        # === 改进的标签编码器 ==
        self.labDescVec = torch.tensor(labDescVec, dtype=torch.float32).to(device)
        self.icdAttn = DeepICDDescCandiAttention(
            inSize=self.emb_dim,
            classNum=classNum,
            labSize=self.lab_dim,
            hdnDropout=hdnDropout,
            attnList=attnList,
            labDescVec=labDescVec
        ).to(device)

        # === 改进的分类头 ===
        # self.fcLinear = MLP(labDescVec.shape[1], 1, [512, 256], dropout=0.3).to(device)
        self.fcLinear = MLP(768, 768, [768], dropout=0.3).to(device)

        # === 简化图编码器 ===
        with open(graph_path, "rb") as f:
            edges_dict = pickle.load(f)
        # 从图中提取所有代码
        graph_nodes = set()
        for (code1, code2) in edges_dict.keys():
            graph_nodes.add(code1)
            graph_nodes.add(code2)
        all_codes = sorted(list(graph_nodes))
        c2ind = {code: idx for idx, code in enumerate(all_codes)}
        # 建立从dataClass索引到图中索引的映射
        self.dataclass_to_graph_mapping = {}
        for i, code in enumerate(dataClass.id2icd):
            if code in c2ind:
                self.dataclass_to_graph_mapping[i] = c2ind[code]
            else:
                self.dataclass_to_graph_mapping[i] = 0

        # 构建 cm2ind：major_code -> 索引
        def extract_major_code(code):
            code_str = str(code)
            parts = code_str.split('_')
            if len(parts) >= 2:
                if len(parts[1]) <= 3:
                    return code_str
                else:
                    return f"{parts[0]}_{parts[1][:3]}"
            return code_str

        major_codes = set()
        for code in all_codes:
            major_code = extract_major_code(code)
            major_codes.add(major_code)
        major_codes = sorted(list(major_codes))
        cm2ind = {code: idx for idx, code in enumerate(major_codes)}
        print(f"[INFO] dataClass.id2icd size = {len(dataClass.id2icd)}")
        print(f"[INFO] 图中代码总数 = {len(all_codes)}")
        print(f"[INFO] c2ind size = {len(c2ind)}")
        print(f"[INFO] cm2ind size = {len(cm2ind)}")
        self.graph_encoder = KGCodeReassign(
            args={
                'attention_dim': 768,
                'use_multihead': 4,
                'edge_dim': 32,
                'rep_dropout': Dropout,
            },
            edges_dict=edges_dict,
            c2ind=c2ind,
            cm2ind=cm2ind
        ).to(device)
        # === 简化图融合机制 ===
        self.w_graph_linear = MLP(
            inSize=768,
            outSize=768,
            hiddenList=[768],
            dropout=0.5,
            actFunc=nn.ReLU
        ).to(device)
        # === 简化的门控融合 ===
        self.gate_fusion = nn.Sequential(
            nn.Linear(1536, 256),  # 2 * 768 = 1536
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)
        # === Module list ===
        self.moduleList = nn.ModuleList([
            self.embedding,
            self.mamba_encoder,
            self.icdAttn,
            self.fcLinear,
            self.graph_encoder,
            self.w_graph_linear,
            self.gate_fusion,
            self.fusion_dropout
        ])

        # 改进的损失函数
        if useCircleLoss:
            self.crition = MultiLabelCircleLoss()
        elif hasattr(self, 'useFocalLoss') and self.useFocalLoss:
            self.crition = MultiLabelFocalLoss()
        else:
            self.crition = nn.MultiLabelSoftMarginLoss()

        self.hdnDropout = hdnDropout
        self.fcDropout = fcDropout
        self.classNum = classNum

    def get_stage1_predictions(self, X):
        """获取FLASH_ICD_FULL模型的预测"""
        if self.stage1_model is not None:
            with torch.no_grad():
                baseline_out = self.stage1_model.calculate_y_logit(X)
                baseline_logits = baseline_out['y_logit']
                return baseline_logits
        else:
            return torch.zeros(X['noteArr'].size(0), self.classNum, device=self.device)

    # def calculate_y_logit(self, X, candidate):
    #     candidate = torch.tensor(candidate, dtype=torch.long, device=self.device) if not torch.is_tensor(
    #         candidate) else candidate.long()
    #
    #     x = self.embedding(X['noteArr'])
    #     x = self.mamba_encoder(x)
    #     x = self.icdAttn(x, candidate=candidate, labDescVec=self.labDescVec)
    #     x = self.fcLinear(x).squeeze(2)
    #
    #     # 只检查NaN，不强制限制范围
    #     if torch.isnan(x).any():
    #         print("Warning: NaN detected in logits")
    #         x = torch.nan_to_num(x, nan=0.0)
    #
    #     return {'y_logit': x}
    def calculate_y_logit(self, X, candidate):
        candidate = torch.tensor(candidate, dtype=torch.long, device=self.device) if not torch.is_tensor(
            candidate) else candidate.long()

        x = self.embedding(X['noteArr'])
        x = self.mamba_encoder(x)
        x = self.icdAttn(x, candidate=candidate, labDescVec=self.labDescVec)
        x = self.fcLinear(x)  # 现在输出是 [B, C, 768]

        # 需要将 768 维投影到 1 维（每个类一个logit）
        # 添加一个投影层，或者使用点积
        # 方案A：使用标签描述向量的点积
        # x: [B, C, 768], self.labDescVec: [classNum, 768]
        # 对每个候选类，计算与对应标签描述向量的相似度
        candidate_desc = self.labDescVec[candidate]  # [B, C, 768]
        x = torch.sum(x * candidate_desc, dim=2)  # [B, C] - 点积得到logit

        # 或者方案B：添加一个轻量级投影层
        # 但这需要修改模型结构，可能影响加载

        if torch.isnan(x).any():
            print("Warning: NaN detected in logits")
            x = torch.nan_to_num(x, nan=0.0)

        return {'y_logit': x}
    def calculate_loss(self, X, Y, candidate):  # ← 必须接受candidate参数

        out = self.calculate_y_logit(X, candidate)
        Y_logit = out['y_logit']

        # 主损失
        main_loss = self.crition(Y_logit, Y)

        # MoE辅助损失
        moe_aux_loss = 0.0
        if hasattr(self.mamba_encoder, 'get_moe_aux_loss'):
            moe_aux_loss = self.mamba_encoder.get_moe_aux_loss()

        total_loss = main_loss + 0.1 * moe_aux_loss
        return total_loss

    def calculate_y_prob(
            self, X, candidate,
            save_analysis=False,
            analysis_save_path="debug_outputs/rare_outputs.jsonl",
            rare_code_set=None,
            id2icd=None
    ):
        # 1. 全集预测（需要MLP层）
        stage1_logits = self.get_stage1_predictions(X)  # 这需要MLP层，导致参数不匹配
        stage1_probs = torch.sigmoid(stage1_logits)

        # 2. 候选集处理
        if not torch.is_tensor(candidate):
            candidate = torch.as_tensor(candidate, device=self.device)
        candidate = candidate.long()

        # 3. 特征提取与候选集预测
        note_repr = self.embedding(X['noteArr'])
        note_repr = self.mamba_encoder(note_repr)
        m = self.icdAttn(note_repr, candidate=candidate, labDescVec=self.labDescVec)
        y_logit_candidate = self.calculate_y_logit(X, candidate)['y_logit']
        p_candidate = torch.sigmoid(y_logit_candidate)

        # 4. 图增强
        if candidate.dim() == 2:
            graph_indices = torch.tensor(
                [[self.dataclass_to_graph_mapping[idx.item()] for idx in row] for row in candidate],
                device=self.device
            )
        else:
            graph_indices = candidate

        # 图编码，支持attention返回
        if save_analysis:
            m_updated, att_info = self.graph_encoder(
                m, m, indices=graph_indices,
                return_attention_weights=save_analysis
            )
        else:
            m_updated = self.graph_encoder(m, m, indices=graph_indices)
            att_info = None

        w_graph = self.w_graph_linear(m_updated)
        logits_graph = torch.einsum("bcd,bcd->bc", m_updated, w_graph)
        p_graph = torch.sigmoid(logits_graph)

        # 5. 门控融合
        gate_input = self.fusion_dropout(torch.cat([m, m_updated], dim=-1))
        B, C, D2 = gate_input.shape
        gate = self.gate_fusion(gate_input.view(-1, D2)).view(B, C)
        p_fused = (1 - gate) * p_candidate + gate * p_graph

        # 6. 置信度mask策略
        stage1_conf = (stage1_probs.gather(1, candidate) - 0.5).abs()
        candidate_conf = (p_fused - 0.5).abs()
        mask = candidate_conf > (stage1_conf + 0.05)
        fused_probs = stage1_probs.clone()
        fused_probs.scatter_(1, candidate, torch.where(mask, p_fused, stage1_probs.gather(1, candidate)))

        # 7. 分析采集
        if save_analysis:
            os.makedirs(os.path.dirname(analysis_save_path), exist_ok=True)
            for b in range(B):
                for c in range(C):
                    label_idx = int(candidate[b][c].cpu().item())
                    icd_code = id2icd[label_idx]
                    if icd_code not in rare_code_set:
                        continue
                    row = {
                        "case_batch_idx": b,
                        "label_idx": label_idx,
                        "icd_code": icd_code,
                        "p_candidate": float(p_candidate[b][c].cpu()),
                        "p_graph": float(p_graph[b][c].cpu()),
                        "gate": float(gate[b][c].cpu()),
                        "p_fused": float(p_fused[b][c].cpu()),
                        "stage1_prob": float(stage1_probs[b][label_idx].cpu()),
                        "final_prob": float(fused_probs[b][label_idx].cpu())
                    }
                    if att_info is not None:
                        row['neighbors'] = att_info  # 需要序列化处理
                    with open(analysis_save_path, 'a') as f:
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')

        return fused_probs
