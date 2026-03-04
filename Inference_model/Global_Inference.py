import torch, time, os, pickle, sys
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import *
from nnLayer import *
from metrics import *
from torch.nn.init import xavier_uniform_ as xavier_uniform
from math import floor
from tqdm import tqdm
from pytorch_lamb import lamb
from sklearn.metrics import roc_auc_score
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from Mamba import *

class FGM():
    def __init__(self, model, emb_name='emb'):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}

    def attack(self, epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
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
        pass

    def calculate_y_logit(self):
        pass

    def train(self, dataClass, batchSize, epoch,
              lr=0.001, momentum=0.9, weightDecay=0.0, stopRounds=10, threshold=0.2, earlyStop=10,
              savePath='model/KAICD', saveRounds=1, isHigherBetter=True, metrics="MiF", report=["ACC", "MiF"],
              optimType='Adam', schedulerType='cosine', warmup_ratio=0.1, dataEnhance=False, dataEnhanceRatio=0.0,
              attackTrain=False, attackLayerName='emb', ema_para=-1):
        dataClass.dataEnhance = dataEnhance
        dataClass.dataEnhanceRatio = dataEnhanceRatio

        if attackTrain:
            self.fgm = FGM(self.moduleList, emb_name=attackLayerName)

        if ema_para > 0:
            self.ema = EMA(self.moduleList, ema_para)
        isBeginEMA = False

        metrictor = Metrictor(dataClass.classNum)
        trainStream = dataClass.random_batch_data_stream(batchSize=batchSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum + batchSize - 1) // batchSize

        num_training_steps = itersPerEpoch * epoch
        num_warmup_steps = int(warmup_ratio * itersPerEpoch * epoch)

        optimizer, schedulerRLR = self.get_optimizer(optimType, schedulerType, lr, weightDecay, momentum,
                                                     num_training_steps, num_warmup_steps)

        mtc, bestMtc, stopSteps = 0.0, -1, 0
        if dataClass.validSampleNum > 0:
            validStream = dataClass.random_batch_data_stream(batchSize=batchSize, type='valid', device=self.device)

        st = time.time()

        for e in range(epoch):
            print(f"Epoch {e + 1} with learning rate {optimizer.state_dict()['param_groups'][0]['lr']:.6f}...")
            print('========== Epoch:%5d ==========' % (e + 1))
            if (ema_para > 0) and (e > 30) and (not isBeginEMA):
                self.ema.register()
                isBeginEMA = True

            pbar = tqdm(range(itersPerEpoch))
            for i in pbar:
                self.to_train_mode()
                X, Y, Candidate = next(trainStream)
                loss = self._train_step(X, Y, optimizer, attackTrain, isBeginEMA)
                if schedulerRLR is not None:
                    schedulerRLR.step()
                pbar.set_description(f"Epoch {e} - Training Loss: {loss.data:.3f}")

            if dataClass.validSampleNum > 0 and (e + 1) % saveRounds == 0:
                if isBeginEMA:
                    self.ema.apply_shadow()
                self.to_eval_mode()
                print('[Total Valid]', end='')
                Y_pre, Y = self.calculate_y_prob_by_iterator(
                    dataClass.one_epoch_batch_data_stream(batchSize=4, type='valid', device=self.device))
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
                        print('The val %s has not improved for more than %d steps in epoch %d, stop training.' % (
                            metrics, earlyStop, e + 1))
                        break
                if isBeginEMA:
                    self.ema.restore()

        self.load("%s.pkl" % savePath, dataClass=dataClass)
        with torch.no_grad():
            print(f'============ Result ============')
            print(f'[Total Valid]', end='')
            Y_pre, Y = self.calculate_y_prob_by_iterator(
                dataClass.one_epoch_batch_data_stream(batchSize=4, type='valid', device=self.device))
            metrictor.set_data(Y_pre, Y, threshold)
            res = metrictor(report)
            print(f'================================')
        torch.cuda.empty_cache()
        return res
    def reset_parameters(self):
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

    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for i, module in enumerate(self.moduleList):
            key = getattr(module, 'name', type(module).__name__ + '_' + str(i))
            if key in parameters:
                module.load_state_dict(parameters[key])
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

    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)['y_logit']
        return torch.sigmoid(Y_pre)

    def calculate_y(self, X, threshold=0.2):
        Y_pre = self.calculate_y_prob(X)
        isONE = Y_pre > threshold
        Y_pre[isONE], Y_pre[~isONE] = 1, 0
        return Y_pre

    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X)
        Y_logit = out['y_logit']

        addLoss = 0.0
        if 'loss' in out: addLoss += out['loss']
        return self.crition(Y_logit, Y) + addLoss

    def calculate_indicator_by_iterator(self, dataStream, classNum, report, threshold):
        metrictor = Metrictor(classNum)
        Y_prob_pre, Y = self.calculate_y_prob_by_iterator(dataStream)
        Metrictor.set_data(Y_prob_pre, Y, threshold)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr, Y_preArr = [], []
        while True:
            try:
                X, Y = next(dataStream)
            except:
                break
            Y_pre, Y = self.calculate_y_prob(X).cpu().data.numpy().astype(np.float16), Y.cpu().data.numpy().astype(
                np.int16)
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

    def _train_step(self, X, Y, optimizer, attackTrain, isBeginEMA):
        loss = self.calculate_loss(X, Y)
        loss.backward()
        if attackTrain:
            self.fgm.attack()
            lossAdv = self.calculate_loss(X, Y)
            lossAdv.backward()
            self.fgm.restore()
        nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        if isBeginEMA:
            self.ema.update()
        optimizer.zero_grad()
        return loss

    def get_optimizer(self, optimType, schedulerType, lr, weightDecay, momentum, num_training_steps, num_warmup_steps):

        # Prepare optimizer and schedule (linear warmup and decay)
        # model_lr={'others':lr, 'flash':0.0007}
        model_lr = {'others': lr}
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        for layer_name in model_lr:
            lr = model_lr[layer_name]
            if layer_name != 'others':  # 设定了特定 lr 的 layer
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
            schedulerRLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 3)
        elif schedulerType == 'None':
            schedulerRLR = None

        return optimizer, schedulerRLR

class DeepLabeler_Contrast(BaseModel):
        def __init__(self, classNum, noteEmbedding, docEmbedding, labDescVec,
                     cnnHiddenSize=64, contextSizeList=[3, 4, 5],
                     docHiddenSize=64, dropout=0.75, device=torch.device('cuda:0'),
                     temp_para=0.1, descSize=768):
            super().__init__()
            self.noteEmbedding = TextEmbedding(noteEmbedding, freeze=False, dropout=dropout, name='noteEmbedding').to(
                device)
            self.docEmbedding = TextEmbedding(docEmbedding, freeze=True, dropout=dropout, name='docEmbedding').to(
                device)
            self.textCNN = TextCNN(noteEmbedding.shape[1], cnnHiddenSize, contextSizeList).to(device)
            self.docFcLinear = MLP(docEmbedding.shape[1], docHiddenSize, name='docFcLinear').to(device)

            # 处理标签描述向量维度
            if labDescVec.shape[1] != descSize:
                print(f"Adjusting labDescVec dimension from {labDescVec.shape[1]} to {descSize}")
                newDescVec = np.zeros((labDescVec.shape[0], descSize))
                min_dim = min(labDescVec.shape[1], descSize)
                newDescVec[:, :min_dim] = labDescVec[:, :min_dim]
                labDescVec = newDescVec

            self.labDescVec = nn.Embedding.from_pretrained(torch.tensor(labDescVec, dtype=torch.float32),
                                                           freeze=False).to(device)
            self.labDescVec.name = 'labDescVec'

            # 文本拼接向量的输出维度
            self.text_out_dim = cnnHiddenSize * len(contextSizeList) + docHiddenSize

            # 加入映射层：将文本向量从 text_out_dim → 768
            self.textProjector = nn.Linear(self.text_out_dim, descSize).to(device)

            # 分类层（可选，不在 similarity 中用到）
            self.fcLinear = MLP(descSize, classNum, name='fcLinear').to(device)

            self.moduleList = nn.ModuleList([
                self.noteEmbedding, self.docEmbedding, self.textCNN,
                self.docFcLinear, self.fcLinear, self.labDescVec, self.textProjector
            ])

            self.device = device
            self.temp_para = temp_para

        def calculate_y_logit(self, input):
            x1 = input['noteArr']#捕获局部语义特征（如关键词、短语级信息）
            x1 = self.noteEmbedding(x1)
            x1 = self.textCNN(x1)  # shape: [B, cnnHiddenSize * len(contextSizeList)]

            x2 = input['noteIdx']#捕获全局文档级特征（如文档类型、来源等元信息）
            x2 = self.docEmbedding(x2)
            x2 = self.docFcLinear(x2)  # shape: [B, docHiddenSize]

            x = torch.cat([x1, x2], dim=1)  # shape: [B, text_out_dim]
            x = self.textProjector(x)  # shape: [B, 768] ← 映射到与标签对齐维度

            # 计算 cosine similarity：x [B, 768] vs labDescVec [num_labels, 768]通过计算 cosine similarity 来评估每个样本 x 与所有标签描述向量之间的相似度
            similarity_distribution = torch.matmul(x, self.labDescVec.weight.T) / (
                    torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) *
                    torch.sqrt(torch.sum(self.labDescVec.weight ** 2, dim=1, keepdim=True)).T
            )

            return {'note_vec': x, 'similarity': similarity_distribution}

        def calculate_loss(self, X, Y):
            out = self.calculate_y_logit(X)
            similarity_distribution = out['similarity']
            return self.loss_with_ContraLearning(similarity_distribution, Y)

            # similarity_distribution: (batch_size, num_labels)，表示样本与各标签的相似度或打分
            # Y: (batch_size, num_labels)，是多标签的 one-hot（多热）向量，表示该样本对应的真实标签。
            #正样本是Y标签空间中和所有值为1的标签相似性较高的标签
        def loss_with_ContraLearning(self, similarity_distribution, Y):
            below_fraction = torch.exp(similarity_distribution / self.temp_para).sum(dim=1, keepdim=True)
            he_1 = (torch.log(torch.exp(similarity_distribution / self.temp_para) / below_fraction) * Y).sum(dim=1,
                                                                                                                 keepdim=True)  ## 只保留真实标签（正样本）位置的对数概率
            cnt_label = Y.sum(dim=1, keepdim=True)
            aw = torch.mean(-1 * he_1 / cnt_label)
            return aw

        def calculate_y_prob(self, X):
            Y_pre = self.calculate_y_logit(X)['similarity']  # -1 ~ 1
            Y_pre = (Y_pre + 1) / 2
            return Y_pre

class EAM_F(BaseModel):
    def __init__(self, classNum, embedding, labDescVec, seqMaxLen, chunk_length, trans_s, attnList=[],
                 embDropout=0.2, hdnDropout=0.2, fcDropout=0.5, numLayers=1,
                 device=torch.device('cuda:0'), useCircleLoss=False, compress=False, useFocalLoss=False):

        super().__init__()
        self.useFocalLoss = useFocalLoss
        emb_dim = embedding.shape[1]
        self.device = device
        self.labDescVec = torch.tensor(labDescVec, dtype=torch.float32).to(device)

        # 文本嵌入模块
        self.embedding = TextEmbedding_1d(embedding, dropout=embDropout).to(device)
        self.mamba_encoder = MambaStack1(
            d_model=emb_dim,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=12,  # 可调层数
            dropout=0.2,  # 可调Dropout
            use_moe_enhancement=True,  # 启用MoE增强
            moe_weight=0.01  # 小权重，仅作为增强模块存在
        ).to(device)
        self.mamba_encoder.name = 'mamba_encoder'
        # ICD 标签描述注意力模块
        self.icdAttn = DeepICDDescAttention(
            emb_dim, classNum, labDescVec.shape[1],
            hdnDropout=hdnDropout, attnList=attnList, labDescVec=labDescVec
        ).to(device)
        # 标签分类线性层
        self.fcLinear = MLP(labDescVec.shape[1], 1, [], dropout=fcDropout).to(device)
        # 模块注册
        self.moduleList = nn.ModuleList([
            self.embedding, self.mamba_encoder, self.icdAttn, self.fcLinear
        ])
        # 损失函数
        if useCircleLoss:
            self.crition = MultiLabelCircleLoss()
        elif hasattr(self, 'useFocalLoss') and self.useFocalLoss:
            self.crition = MultiLabelFocalLoss()
        else:
            self.crition = nn.MultiLabelSoftMarginLoss()
    def calculate_y_logit(self, input):
        x = input['noteArr']  # 输入 shape: [B, L]

        if torch.cuda.device_count() > 1:
            x = nn.parallel.data_parallel(self.embedding, x)
            x = nn.parallel.data_parallel(self.mamba_encoder, x)
            x = nn.parallel.data_parallel(self.icdAttn, x)
            x = nn.parallel.data_parallel(self.fcLinear, x).squeeze(dim=2)
        else:
            x = self.embedding(x)
            x = self.mamba_encoder(x)                  # => [B, L, D]
            x = self.icdAttn(x)                         # => [B, C, D']
            x = self.fcLinear(x).squeeze(dim=2)        # => [B, C]
        return {'y_logit': x}
    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X)
        Y_logit = out['y_logit']

        # 主损失
        main_loss = self.crition(Y_logit, Y)

        # MoE辅助损失
        moe_aux_loss = 0.0
        if hasattr(self.mamba_encoder, 'get_moe_aux_loss'):
            moe_aux_loss = self.mamba_encoder.get_moe_aux_loss()

        # 总损失 = 主损失 + MoE辅助损失
        total_loss = main_loss + 0.1 * moe_aux_loss
        return total_loss

