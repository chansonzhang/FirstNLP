# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 11/28/2018


import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline_Model(nn.Module):
    def __init__(self, config, embedding_pre):
        super(Baseline_Model, self).__init__()

        # Set hyper parameters
        self.word_emb_dim = config['word_emb_dim']
        self.pos_emb_dim = config['pos_emb_dim']
        self.pos_tag_dim = config['pos_tag_dim']
        self.pos_tag_vocab = config['pos_tag_vocab']
        self.e1_head_dim = config['e1_head_dim']
        self.e1_head_vocab = config['e1_head_vocab']
        self.e2_head_dim = config['e2_head_dim']
        self.e2_head_vocab = config['e2_head_vocab']
        self.hidden_dim = config['hidden_dim']  # 隐藏单元个数
        self.output_dim = config['output_dim']  # 输出维度
        self.input_dim = config['word_emb_dim']  # 输入维度
        # self.embedding_pre = embedding_pre
        self.MAX_POS = config['MAX_POS']  # 最大位置值

        # Set options and other parameters
        # self.use_gpu = use_gpu#是否使用GPU
        self.word_vocab = config['word_vocab']  # 单词数，词表大小
        # self.label_vocab = label_vocab#分类表词表
        # self.pos_vocab = pos_vocab

        # Free parameters for the model模型的自由参数
        # Initialize embeddings (Word and Position embeddings)初始化词和位置嵌入
        # self.word_emb = nn.Embedding(len(self.word_vocab), self.word_emb_dim).cuda()
        # self.word_emb = nn.Embedding(self.word_vocab, self.word_emb_dim)
        self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
        self.pos_tag_emb = nn.Embedding(self.pos_tag_vocab, self.pos_tag_dim)
        self.e1_head_emb = nn.Embedding(self.e1_head_vocab, self.e1_head_dim)
        self.e2_head_emb = nn.Embedding(self.e2_head_vocab, self.e2_head_dim)
        # self.pos1_emb = nn.Embedding(self.MAX_POS*2+1, self.pos_emb_dim).cuda()
        self.pos1_emb = nn.Embedding(self.MAX_POS * 2 + 1, self.pos_emb_dim)
        # self.pos1_emb.weight.data.uniform_(-0.0, 0.0)
        # self.pos2_emb = nn.Embedding(self.MAX_POS*2+1, self.pos_emb_dim).cuda()
        self.pos2_emb = nn.Embedding(self.MAX_POS * 2 + 1, self.pos_emb_dim)
        # self.pos2_emb.weight.data.uniform_(-0.0, 0.0)

        # Initialize LSTM parameters ()初始化BiGRU的参数
        # self.rnn = nn.GRU(self.input_dim, hidden_dim, bidirectional=True, batch_first = True).cuda()
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True, batch_first=True)

        # Initialize Attention parameters ()初始化Attention的参数
        # self.attention_hidden = nn.Linear(hidden_dim * 2  +  self.pos_emb_dim * 2, hidden_dim,bias=False).cuda()
        self.attention_hidden = nn.Linear(
            self.hidden_dim * 2 + self.pos_tag_dim +  self.e1_head_dim +  self.e2_head_dim + self.pos_emb_dim * 2, hidden_dim,
            bias=False)
        # self.attention = nn.Linear(hidden_dim, 1, bias=False).cuda()
        self.attention = nn.Linear( self.hidden_dim, 1, bias=False)

        # Initialize Classifier parameters ()初始化分类器参数
        # self.classifier_hidden = nn.Linear(hidden_dim * 2, hidden_dim).cuda()
        # self.classifier = nn.Linear(hidden_dim * 2, output_dim).cuda()
        self.classifier = nn.Linear( self.hidden_dim * 2,  self.output_dim)


    def forward(self, sentence, pos_tag, e1_head, e2_head, pos1, pos2, is_train=True):
        # print("sentence.shape:",sentence.shape)
        # print("pos1.shape:",pos1.shape)
        # print("pos2.shape:",pos2.shape)
        word_embeddings = self.word_emb(sentence)
        pos_tag_embeddings = self.pos_tag_emb(pos_tag)
        e1_head_embeddings = self.e1_head_emb(e1_head)
        e2_head_embeddings = self.e1_head_emb(e2_head)
        pos1_embeddings = self.pos1_emb(pos1)
        pos2_embeddings = self.pos2_emb(pos2)

        # LSTM layer
        # X = torch.cat((word_embeddings,pos_tag_embeddings,e1_head_embeddings,e2_head_embeddings,pos1_embeddings, pos2_embeddings),-1)
        X = word_embeddings
        # print("word_embeddings:",X[:2])
        # print("pos1_embeddings:",pos1_embeddings[:2])
        # print("pos2_embeddings:",pos2_embeddings[:2])
        X = F.dropout(X, p=0.5)  # 将词嵌入层Droupout下
        hiddens, for_output = self.rnn(X)
        # print("output.shape:",hiddens.shape)
        # rev_hiddens, rev_output = self.rev_lstm(X)

        # Self Attentive layer自注意力层
        att_input = torch.cat(
            (hiddens,
             pos_tag_embeddings,
             e1_head_embeddings,
             e2_head_embeddings,
             pos1_embeddings,
             pos2_embeddings),
            -1)  # 拼接
        # print("att_input.shape:",att_input.shape)
        att_input = F.dropout(att_input, p=0.5)  # 对拼接后的层dropout
        att_hidden = torch.tanh(self.attention_hidden(att_input))  # feed自注意力层
        # print("att_hidden.shape:", att_hidden.shape)
        # att_hidden = F.dropout(att_hidden, p=0.5)#对att_hiddendropout下
        att_scores = self.attention(att_hidden)  # 得到注意力分数

        # print("att_scores.shape:",att_scores.shape)
        # print("att_scores.shape:", att_scores.shape)
        attention_distrib = F.softmax(att_scores, dim=1)  # 对分数进行归一化,a=softmax(ws2*tanh(ws1*H.transpose))
        # print("attention_distrib.shape:",attention_distrib.shape)
        # hiddens = F.dropout(hiddens, p=0.5)#hiddens dropout下
        context_vector = torch.sum(hiddens * attention_distrib, dim=1)  # 得到背景向量context_vector,加权求和
        # print("context_vector.shape:",context_vector.shape)
        # Classifier
        # context_hidden = self.classifier_hidden(context_vector)
        # context_hidden = F.dropout(context_hidden, p=0.5, training=is_train)

        finals = F.softmax(self.classifier(context_vector), dim=1)  # 将背景向量送入分类器进行分类
        # print("finals.shape",finals.shape)
        return finals


# 依存句法分析
arcs = parser.parse(words, postags)  # 句法分析
for arc in arcs:
    # arc.relation = ' '.join(arc.relation)
    parsers.append(arc.head)
    parsers_r.append(arc.relation)
e1_head.append(parsers[index1])
e2_head.append(parsers[index2])

e1_relation.append(parsers_r[index1])
e2_relation.append(parsers_r[index2])