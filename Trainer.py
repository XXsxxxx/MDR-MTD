
# ######################全脑
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.ms_ast_gcn import MultiStageModel
# import torch.optim as optim

# class backbone_network(nn.Module):
#     def __init__(self, opt):
#         super(backbone_network,self).__init__()
#         self.opt = opt
#         self.trainer = MultiStageModel(opt, opt['num_stages'])

#         # Emotion Classifier        
#         self.all_output_layer = nn.Sequential(
#             nn.Linear(128*opt['dim_model'], 2048),  # 从 1280 映射到 1024
#             nn.ReLU(),
#             nn.Linear(2048, 1024),
#             # nn.Dropout(opt['dropout']),
#             nn.Linear(1024, 512),   # 从 1024 映射到 512
#             nn.ReLU(),
#             nn.Linear(512, 64),     # 从 512 映射到 64
#             nn.Linear(64, opt['num_class'])  # 从 64 映射到类别数
#         )

#         # Domain Classifier
#         self.DomainDiscriminator = nn.Sequential(
#             nn.Linear(128 * opt['dim_model'], 1024),
#             nn.ReLU(),
#             nn.Dropout(opt['dropout']),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(opt['dropout']),
#             nn.Linear(512, 64),  
#             nn.ReLU(),
#             nn.Dropout(opt['dropout']),
#             nn.Linear(64, 2)   # 输出 2 表示两个域标签：源域和目标域
#         )
        
#         self.loss_function = nn.CrossEntropyLoss().cuda()
#         self.domain_loss_function = nn.CrossEntropyLoss().cuda()  # 域鉴别器损失

#         # 优化器
#         self.parameters = [p for p in self.trainer.parameters() if p.requires_grad]
#         self.optimizer = optim.Adam(self.parameters, lr=opt['lr'], weight_decay=opt['weight_decay'])
#         self.domain_optimizer = optim.Adam(self.DomainDiscriminator.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])


#     def train (self, train_x, train_adj, train_y, test_x, test_adj):

#         # 设置训练模式
#         self.trainer.train()
#         self.DomainDiscriminator.train()

#         # 优化器梯度清零
#         self.optimizer.zero_grad()
#         self.domain_optimizer.zero_grad()

#         # 获取源域（训练集）和目标域（测试集）的特征
#         train_outs, domain_outs, mmd_loss = self.trainer(train_x, train_adj)
#         test_outs, test_domain_outs, __ = self.trainer(test_x, test_adj)

#         # 情绪分类损失
#         reshape_out = train_outs.reshape(train_outs.shape[0], train_outs.shape[1], -1)
#         processed_outs = self.all_output_layer(reshape_out.view(-1, reshape_out.shape[2]))
#         last_out = processed_outs.view(reshape_out.shape[0], reshape_out.shape[1], -1)
#         Cls_loss = sum(self.loss_function(out, train_y.long()) for out in last_out)

#         # 域鉴别器损失
#         dom_features = torch.cat((domain_outs, test_domain_outs), dim=1)  # 拼接训练和测试特征
#         reshape_out = dom_features.reshape(dom_features.shape[0], dom_features.shape[1], -1)
#         domain_labels = torch.cat((torch.zeros(domain_outs.size(1)), torch.ones(test_domain_outs.size(1)))).long().cuda()  # 0为源域，1为目标域
#         domain_preds = self.DomainDiscriminator(reshape_out.view(-1, reshape_out.shape[2]))
#         domain_preds = domain_preds.view(dom_features.shape[0], dom_features.shape[1], -1)
#         domain_loss = sum(self.domain_loss_function(out, domain_labels) for out in domain_preds)

#         # 总损失
#         loss = Cls_loss + self.opt['Lamda'] * mmd_loss + self.opt['Domain_Lambda'] * domain_loss
#         # loss = Cls_loss + self.opt['Lamda'] * mmd_loss

#         # 反向传播与更新
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.parameters, self.opt['max_grad_norm'])
#         self.optimizer.step()
#         self.domain_optimizer.step()

#         return last_out, loss


#     def predict(self, test_x, test_adj):

#         self.trainer.eval()
#         outs = self.trainer(test_x, test_adj)
#         reshape_out = outs.reshape(outs.shape[0], outs.shape[1], -1)   ###2*40*1280
#         processed_outs = self.all_output_layer(reshape_out.view(-1, reshape_out.shape[2]))
#         # 恢复为 (2, 40, num_classes)
#         last_out = processed_outs.view(reshape_out.shape[0], reshape_out.shape[1], -1)  ###2*40*class
#         return last_out
    

###############################半脑+域自适应

import torch
import torch.nn as nn
# import torch.nn.functional as F
from t1_t2_s1 import ST
import torch.optim as optim

class backbone_network(nn.Module):
    def __init__(self, opt):
        super(backbone_network, self).__init__()
        self.opt = opt
        self.trainer = ST(opt)

        # Emotion Classifier
        self.all_output_layer = nn.Sequential(
            nn.Linear(128 * opt['dim_model'], 1024),
            nn.ReLU(),
            nn.Dropout(opt['dropout']),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Linear(64, opt['num_class'])
        )


        # self.classifier = nn.Sequential(
        #     nn.Linear(gpt_n_embd * num_chunks, gpt_n_embd * num_chunks // 2),
        #     nn.ReLU(),
        #     nn.Linear(gpt_n_embd * num_chunks // 2, gpt_n_embd * num_chunks // 4),
        #     nn.ReLU(),
        #     nn.Linear(gpt_n_embd * num_chunks // 4, gpt_n_embd),
        #     nn.ReLU(),
        #     nn.Linear(gpt_n_embd, opt['num_class'])
        # )
        self.loss_function = nn.CrossEntropyLoss().cuda()
        self.domain_loss_function = nn.CrossEntropyLoss().cuda()  # 域鉴别器损失

        # 优化器
        self.parameters = [p for p in self.trainer.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(self.parameters, lr=opt['lr'], weight_decay=opt['weight_decay'])


    def train(self, train_x, train_y):
        # 设置训练模式
        self.trainer.train()

        # 优化器梯度清零
        self.optimizer.zero_grad()


        # 获取源域（训练集）和目标域（测试集）的特征
        logits_s, reversed_features, loss_re,loss_t1,loss_t2,loss_s,loss_dist1 = self.trainer(train_x,train_y)



        # 总损失
        loss = (self.opt['cls_lambda'] * loss_s +
                self.opt['t1_lambda'] * loss_t1 +
                self.opt['t2_lambda'] * loss_t2 +
                self.opt['re_lambda'] * loss_re +
                self.opt['dist_lambda'] * loss_dist1)

        # 反向传播与更新
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.opt['max_grad_norm'])
        self.optimizer.step()

        return logits_s, loss

    def predict(self, test_x, test_y):
        self.trainer.eval()

        outs = self.trainer(test_x, test_y)

        return outs

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, is_ca=False):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        if is_ca:
            loss = (nn.KLDivLoss(reduction='none')(p_s, p_t) * (self.T ** 2)).sum(-1)
        else:
            loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T ** 2)
        return loss