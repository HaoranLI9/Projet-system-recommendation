from typing_extensions import get_origin
import torch.nn as nn
import torch
from utils.util import conf_din


class Dice(nn.Module):
    
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1, )))
        
    def forward(self, x):
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        norm_x = (x - avg) / std
        p = torch.sigmoid(norm_x)

        return x.mul(p) + self.alpha * x.mul(1 - p)

class attention_unit(nn.Module):
    def __init__(self, num_features):
        super(attention_unit, self).__init__()
        l1_dim = conf_din['runtime']['att_mlp_lin1']
        l2_dim = conf_din['runtime']['att_mlp_lin2']
        self.mlp = nn.Sequential(
            nn.Linear(num_features, l1_dim),
            Dice(),
            nn.Linear(l1_dim, l2_dim),
            Dice(),
            nn.Linear(l2_dim, 1)
        )

    def forward(self, query, facts, mask):
        '''
            param:
                query: candidate_item's embedding, (batch_size, 2H)
                facts: user behaviour sequence, (seqLen, batch_size, 2H)
                mask: mask for padded sequence, (batch_size, true_length)

            return:
                attention_output: weights for facts, (batch_size, seqLen)
        '''
        # inner produect, concat
        #print(query.shape, facts.shape, mask.shape)
        #torch.Size([1024, 256]) torch.Size([406, 1024, 256]) torch.Size([406, 1024])
        query2 = query.tile((1, facts.size(0))).reshape(facts.size()) # reshape query to the size of facts
        #print(query2.shape, facts.shape)
        input = torch.cat([query2, facts, query2*facts, query2-facts], dim=-1)   # input.size() == (seqLen, batch_size, 2H*4=8H)
        #print(input.shape, query.shape) #input.size - 4 * query.size
        #print(input.size())
        #assert input.size(-1) == 8*8
        output = self.mlp(input)
        #print(output.shape)([254, 1024, 1])
        
        #print(output.size()) # (seqLen, batch_size, 1)
        output = output.squeeze(-1) # (seqLen, batch_size)
        #print(mask.shape, output.shape)
        #print(mask)
        #print(output)
        masked = torch.where(mask, output, torch.cuda.FloatTensor([-2**32 + 1]))  # mask paddings, use small negative number to make exp value 0
        #print(masked)
        assert 1
        normalized = torch.softmax(masked,dim=0)    # normalize by softmax for every sequence
        # weighted sum with facts
        res = normalized.unsqueeze(-1) * facts
        #print(res.shape) #torch.Size([430, 1024, 256])
        return res

class MultiLayerPerceptron(nn.Module):

    def __init__(self, layer, batch_norm=True):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        input_size = layer[0]
        for output_size in layer[1: -1]:
            layers.append(nn.Linear(input_size, output_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.Dropout(p=0.5))
            layers.append(Dice())
            input_size = output_size
        layers.append(nn.Linear(input_size, layer[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class Base(nn.Module):
    def __init__(self, feature_size_dict):
        '''
                feature_size_dict = {
                    'unixReviewTime' : len(unixReviewTime_buckets),
                    'itempos' : seq_len_max,
                    'itemId': len(item2idx_dict) + 1,
                    'itemCat': seq_len_max,
                }
        '''
        super().__init__()
        emb_dim = conf_din['runtime']['emb_dim']
        h = 6 * emb_dim #48
        feature_size_dict['itemCat'] = 801
        #feature_size_dict['itemId'] = 
        self.embs = nn.ModuleDict(
            [(feature_name, nn.Embedding(size, embedding_dim=emb_dim)) for feature_name, size in feature_size_dict.items()])
        #print(self.embs)
        assert 1
        """
        self.mlp = nn.Sequential(
            nn.Linear(h, 200),
            nn.BatchNorm1d(200),
            Dice(),

            nn.Linear(200, 80),
            nn.BatchNorm1d(80),
            Dice(),

            nn.Linear(80, 2),
            nn.Sigmoid()
        )
        """
        self.mlp = MultiLayerPerceptron([h, 200, 80, 2])

    def forward(self, context_unixReviewTime,
                context_itempos,
                behaviour_itemId,
                behaviour_itemCat,
                candidate_itemId,
                candidate_itemCat):
        #print(context_unixReviewTime.size())
        #behaviour_group = [self.embs['itemId'](behaviour_itemId), self.embs['itemId'](behaviour_itemCat)]
        behaviour_group = [self.embs['itemId'](behaviour_itemId), self.embs['itemCat'](behaviour_itemCat)]#, self.embs['itemId'](behaviour_itemId)]
        context_group = [self.embs['unixReviewTime'](context_unixReviewTime), self.embs['itempos'](context_itempos)]
        candidate_group = [self.embs['itemId'](candidate_itemId), self.embs['itemCat'](candidate_itemCat)]
        groups = [behaviour_group, context_group, candidate_group]
        # concatenation over group, dim=-1 => the last dimension.
        # (seqlen, batch_size, dim=128) => (seqlen, batch_size, dim=128*num_features), with num_features=2
        groups = list(map(lambda group: torch.cat(group, dim=-1), groups))
        # perform sum pooling over behaviours and context_features
        # (seqlen, batch_size, dim=256) => (batch_size, dim=256)
        groups[:-1] = list(map(lambda tensor : torch.sum(tensor, dim=0), groups[:-1]))
        # concat all & flatten
        # (batch_size, dim=256) => (batch_size, dim=256*n=768), n = len(groups) = 3
        input = torch.cat(groups, dim=-1)
        #assert input.size() == (32, 768)
        #print("aaa", input.size())
        #input = self.bn(input)
        #print(input.shape)
        #print("input", input.shape)
        output =  self.mlp(input)
        output = torch.sigmoid(output)
        return output

class DIN(nn.Module):
    def __init__(self, feature_size_dict):
        '''
                feature_size_dict = {
                    'unixReviewTime' : len(unixReviewTime_buckets),
                    'itempos' : seq_len_max + 1,
                    'itemId': len(item2idx_dict) + 1,
                    'itemCat': seq_len_max + 1,
                }

                seq_lengths
        '''
        super().__init__()
        emb_dim = conf_din['runtime']['emb_dim']
        feature_size_dict['itemCat'] = 801
        h = 4 * emb_dim
        self.embs = nn.ModuleDict(
            [(feature_name, nn.Embedding(size, embedding_dim=h)) for feature_name, size in feature_size_dict.items()])
        h1_dim = conf_din['runtime']['din_mlp_lin1']
        h2_dim = conf_din['runtime']['din_mlp_lin2']
        self.attention_layer = attention_unit(8*h) if conf_din['runtime']['use_attention'] else None
        self.mlp = MultiLayerPerceptron([4*h, 200, 80, 2])
        #self.bn = nn.BatchNorm1d(4*h)

    def forward(self, context_unixReviewTime,
                context_itempos,
                behaviour_itemId,
                behaviour_itemCat,
                candidate_itemId,
                candidate_itemCat,
                mask):
        #print(context_unixReviewTime.size())
        behaviour_group = [self.embs['itemId'](behaviour_itemId), self.embs['itemCat'](behaviour_itemCat)]
        #context_group = [self.embs['unixReviewTime'](context_unixReviewTime), self.embs['itempos'](context_itempos)]
        candidate_group = [self.embs['itemId'](candidate_itemId), self.embs['itemCat'](candidate_itemCat)]
        groups = [behaviour_group, candidate_group]
        # concatenation over group, dim=-1 => the last dimension.
        # (seqlen, batch_size, H) => (seqlen, batch_size, 2H)
        groups = list(map(lambda group: torch.cat(group, dim=-1), groups))
        if self.attention_layer:
            attention_output = self.attention_layer(
                groups[-1],
                groups[0],
                mask
            )
            groups = [attention_output, groups[-1]]
        # perform sum pooling over behaviours and context_features
        # (seqlen, batch_size, 2H) => (batch_size, 2H)
        groups[:-1] = list(map(lambda tensor : torch.sum(tensor, dim=0), groups[:-1]))
        # concat all & flatten
        # (batch_size, 2H) => (batch_size, 6H)
        input = torch.cat(groups, dim=-1)
        #print("aaa", input.shape)
        #input = self.mlp(input)
        #print("bbb", input.shape)
        output =  self.mlp(input)
        output = torch.sigmoid(output)
        return output