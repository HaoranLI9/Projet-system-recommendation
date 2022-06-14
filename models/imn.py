import torch
import torch.nn as nn

class Embedding_Layer(nn.Module):
    def __init__(self, embedded_size, max_position, max_time, max_item):  ## max_time & max_item
        super(Embedding_Layer, self).__init__()
        self.embed_item = nn.Embedding(max_item+1, embedded_size)
        self.embed_time = nn.Embedding(max_time+1, embedded_size)
        self.embed_position = nn.Embedding(max_position+1, embedded_size)

    def forward(self, seq, time=None, position=None):
        embed_seq = self.embed_item(seq)
        if time is not None and position is not None:
            embed_time = self.embed_time(time)
            embed_position = self.embed_position(position)
            embed_seq = embed_seq + embed_time + embed_position

        return embed_seq

class Multiway_Attention(nn.Module):
    def __init__(self, input_size, output_size): ## input_size (embedding_size)
        """
        seq : batch * length * embedding_size
        target : embedding_size
        memory : embedding_size
        """
        super(Multiway_Attention, self).__init__()
        self.linear1 = nn.Linear(4*input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq, target, memory):
        #target = target.view(-1)
        #memory = memory.view(-1)
        target = target.unsqueeze(1).repeat(1,seq.size(1),1)
        memory = memory.unsqueeze(1).repeat(1, seq.size(1), 1)
        alpha = torch.cat([seq-memory, seq-target, seq*target, seq*target], dim=-1)

        res = self.sigmoid(self.linear2(self.sigmoid(self.linear1(alpha))))

        #print(res.size())
        return res

class Interactive_Memory_Update(nn.Module):
    def __init__(self, N, input_size): ## input_size(embedding_size)
        super(Interactive_Memory_Update, self).__init__()
        self.N = N
        self.gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        self.ma = Multiway_Attention(input_size=input_size, output_size=input_size)

    def forward(self, seq, target):
        m = target.detach().clone()
        for _ in range(self.N):
            alpha = self.ma(seq, target, m)
            user_interest = torch.sum(alpha*seq, 1) # user_interest : batch * embedding_size
            #print(user_interest.size(), m.size())
            _, m = self.gru(user_interest.unsqueeze(1), m.unsqueeze(0))
            m = m.squeeze(0)

        return m

class Memory_Enhancement(nn.Module):
    def __init__(self, t, input_size): ## input_size(embedding_size)
        super(Memory_Enhancement, self).__init__()
        self.t = t
        output_size = input_size
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)

    def forward(self, target, memory):
        """
        u : batch * embedding_size
        """
        u = memory
        for _ in range(self.t):
            input = torch.cat([self.linear(u).unsqueeze(1), target.unsqueeze(1)], dim=1)
            #print('ici', input.size(), u.size())
            _, u = self.gru(input, u.unsqueeze(0))
            u = u.squeeze(0)

        return u

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 2)
        self.active = nn.ReLU()
        self.sm = nn.Softmax(dim=-1)
        self.model = nn.Sequential(
                                    self.linear1,
                                    self.active,
                                    self.linear2,
                                    self.active,
                                    self.linear3,
                                    self.active,
                                    self.linear4,
                                    self.sm
                                )

    def forward(self, *args):
        tensor_input = torch.cat(args)

        return self.model(tensor_input)


class IMN(nn.Module):
    def __init__(self, embedded_size, max_position, max_time, max_item, N, t):
        """
        N : nombre d'itérations de memory update process
        t : nombre d'itérations de memory enhancement process
        """
        super(IMN, self).__init__()
        self.N = N
        self.t = t
        self.embed = Embedding_Layer(embedded_size, max_position, max_time, max_item)
        self.memory_update = Interactive_Memory_Update(N, embedded_size)
        self.memory_enhancement = Memory_Enhancement(t, embedded_size)
        self.mlp = MLP(2 * embedded_size)

    def forward(self, seq, target, time=None, position=None):
        """
        target : l'item après le dernier de chaque seq
        """
        embed_seq = self.embed(seq, time, position) #ToDo: Test
        embed_target = self.embed(target)
        memory = self.memory_update(embed_seq, embed_target)#ToDo: Test
        #print(memory.size())
        u = self.memory_enhancement(embed_target, memory)# u : batch_size * embedded_size #ToDo: Test
        input_vector = torch.cat([u, embed_target], dim=1)
        output = self.mlp(input_vector)

        return output




















