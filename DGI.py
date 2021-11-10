import torch
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn.inits import reset, uniform

EPS = 1e-15

# def cluster(data, k, temp, num_iter, init = None, cluster_temp=5):
#     '''
#     pytorch (differentiable) implementation of soft k-means clustering.
#     '''
#     #normalize x so it lies on the unit sphere
#     data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
#     #use kmeans++ initialization if nothing is provided
#     if init is None:
#         data_np = data.detach().numpy()
#         norm = (data_np**2).sum(axis=1)
#         init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
#         init = torch.tensor(init, requires_grad=True)
#         if num_iter == 0: return init
#     #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#     mu = init.to(device)
#     n = data.shape[0]
#     d = data.shape[1]
# #    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
#     for t in range(num_iter):
#         #get distances between all data points and cluster centers
# #        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
#         dist = data @ mu.t()
#         #cluster responsibilities via softmax
#         r = torch.softmax(cluster_temp*dist, 1)
#         #total responsibility of each cluster
#         cluster_r = r.sum(dim=0)
#         #mean of points in each cluster weighted by responsibility
#         cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
#         #update cluster means
#         new_mu = torch.diag(1/cluster_r) @ cluster_mean
#         mu = new_mu
#     dist = data @ mu.t()
#     r = torch.softmax(cluster_temp*dist, 1)
#     return mu, r, dist

class DeepGraphInfomax(torch.nn.Module):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """

    def __init__(self, hidden_channels, encoder, summary, corruption, args, cluster):
        super(DeepGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.reset_parameters()
        self.K = args.K
        self.cluster_temp = args.clustertemp
        self.init = torch.rand(self.K,hidden_channels)
        self.cluster = cluster
        
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(self, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(*args, **kwargs)
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z)
        num_iter = 1
        mu_init, _, _ = self.cluster(pos_z, self.K, 1, num_iter, self.cluster_temp, self.init)
        mu, r, dist = self.cluster(pos_z, self.K, 1, 1, self.cluster_temp, mu_init.detach().clone())
        return pos_z, neg_z, summary, mu, r, dist

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        #print("shape", z.shape,summary.shape)
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutal information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        
        # print('pos_loss = {}, neg_loss = {}'.format(pos_loss, neg_loss))
        # bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
        # modularity = (1./bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()
        return pos_loss + neg_loss #+ modularity

    def comm_loss(self,pos_z,mu):
        return -torch.log(self.discriminate(pos_z,self.summary(mu),sigmoid=True) + EPS).mean()

    def modularity(self, mu, r, embeds, dist, bin_adj, mod, args):
        bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
        return (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()

    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
