from torchtools import *

class EmbeddingImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                              out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        output_data = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        return self.layer_last(output_data.view(output_data.size(0), -1))

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim=48,dropout=0.1):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self,A_new, A_init, X):
        X = self.drop(X)
        X1 = torch.bmm(A_new, X)
        X2 = torch.bmm(A_init,X)
        X = torch.cat([X1,X2],dim=-1)
        X = self.proj(X)
        return X


class MLP(nn.Module):
    def __init__(self,in_dim,hidden=96,ratio=[2,2,1,1]):
        super(MLP, self).__init__()
        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_dim,
                                              out_channels=hidden*ratio[0],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[0]),
                                    nn.LeakyReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=hidden*ratio[0],
                                              out_channels=hidden*ratio[1],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[1]),
                                    nn.LeakyReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[1],
                                              out_channels=hidden * ratio[2],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[2]),
                                    nn.LeakyReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[2],
                                              out_channels=hidden * ratio[3],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[3]),
                                    nn.LeakyReLU())
        self.conv_last = nn.Conv2d(in_channels=hidden * ratio[3],
                                              out_channels=1,
                                              kernel_size=1)
    def forward(self,X):
        # compute abs(x_i, x_j)
        x_i = X.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        A_new = self.conv_last(self.conv_4(self.conv_3(self.conv_2(self.conv_1(x_ij))))).squeeze(1)

        A_new = F.softmax(A_new,dim=-1)

        return A_new

class GNN(nn.Module):
    def __init__(self,
                 in_features,
                 num_classes,
                 hid_dim=48,
                 num_layers=2,
                 dropout=0.1):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        for l in range(num_layers):
            in_dim = in_features + hid_dim*l + tt.arg.num_ways
            node2edge_net = MLP(in_dim=in_dim)

            edge2node_net = GCN(in_dim=in_dim*2,out_dim=hid_dim)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)
        in_dim = in_features + hid_dim*num_layers + tt.arg.num_ways
        last_net_mlp = MLP(in_dim)
        last_net_gcn = GCN(in_dim*2,num_classes)
        self.add_module('last_net_mlp', last_net_mlp)
        self.add_module('last_net_gcn', last_net_gcn)

    def forward(self, X, A_init):

        for l in range(self.num_layers):
            A_new = self._modules['node2edge_net{}'.format(l)](X)
            X_new = self._modules['edge2node_net{}'.format(l)](A_new, A_init, X)
            X = torch.cat([X,X_new],dim=-1)
        A_new = self._modules['last_net_mlp'.format(l)](X)
        out = self._modules['last_net_gcn'.format(l)](A_new,A_init, X)
        out = F.log_softmax(out,dim=-1)

        return out

