from model import Encoder, corruption, Summarizer, cluster_net
import torch
import argparse
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from DGI import DeepGraphInfomax
from sklearn import cluster
import evaluation
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate.')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='Cora',
                    help='which network to load')
parser.add_argument('--K', type=int, default=7,
                    help='How many partitions')
parser.add_argument('--clustertemp', type=float, default=30,
                    help='how hard to make the softmax for the cluster assignments')
parser.add_argument('--train_iters', type=int, default=1001,
                    help='number of training iterations')
parser.add_argument('--num_cluster_iter', type=int, default=1,
                    help='number of iterations for clustering')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def make_adj(x):
    n = 0
    if args.dataset=='Cora':
        n = 2708
    if args.dataset=='Citeseer':
        n = 3327
    if args.dataset=='Pubmed':
        n = 19717
    adj = np.zeros((n,n),dtype=float)
    for i in range(0,len(x[0])):
        adj[x[0][i]][x[1][i]] = 1
    return adj

def make_modularity_matrix(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod

# Loading data
dataset = Planetoid(root='/tmp/'+args.dataset, name=args.dataset)
#dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
#dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
# dataset.transform = T.NormalizeFeatures()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
adj_all = torch.from_numpy(make_adj(dataset[0].edge_index.numpy())).float()
test_object = make_modularity_matrix(adj_all)
data = dataset[0].to(device)


max_nmi = 0
max_ac = 0




# Setting up the model and optimizer
hidden_size = args.hidden
model = DeepGraphInfomax(
    hidden_channels=hidden_size, encoder=Encoder(dataset.num_features, hidden_size),
    summary=Summarizer(),
    corruption=corruption,
    args=args,
    cluster=cluster_net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)


def count(label):
    cnt = [0] * 7
    for i  in label:
        cnt[i] += 1
    print(cnt)

def result(pred, labels):
    nmi = evaluation.NMI_helper(pred, labels)
    ac = evaluation.matched_ac(pred, labels)
    f1 = evaluation.cal_F_score(pred, labels)[0]
    return nmi,ac,f1

def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary, mu, r, dist = model(data.x, data.edge_index)
    dgi_loss = model.loss(pos_z, neg_z, summary)
    modularity_loss = model.modularity(mu,r,pos_z,dist,adj_all,test_object, args)
    comm_loss = model.comm_loss(pos_z,mu)
    #loss = -modularity_loss + 5 * dgi_loss + comm_loss
    loss = dgi_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model):
    model.eval()
    
    with torch.no_grad():
        node_emb, _, _, _, r, _ = model(data.x, data.edge_index)
    r_assign = r.argmax(dim=1)
    cluster_model = cluster.KMeans(n_clusters=dataset.num_classes)
    cluster_model.fit(node_emb.cpu())
    pred = cluster_model.labels_
    labels = data.y.cpu().numpy()
    
    print('label is:')
    count(labels)
    print('result of kmeans is:')
    count(pred)
    nmi,ac,f1 = result(pred, labels)
    r_nmi,r_ac,r_f1 = result(r_assign.numpy(),labels)
    print("dgi_metrics: ",nmi,ac,f1)
    print("clusternet_METRICS: ",r_nmi,r_ac,r_f1)
    return max(nmi,r_nmi),max(ac,r_ac)
    '''
    print('result of PCA is:')
    from sklearn.decomposition import PCA
    estimator = PCA(n_components=dataset.num_classes)
    PCA_score = torch.Tensor(estimator.fit_transform(node_emb.cpu()))
    PCA_pred = torch.max(PCA_score, dim=-1).indices.numpy()
    count(PCA_pred)
    result(PCA_pred, labels)
    '''
    # print ("NMI " + str(evaluation.NMI_helper(pred, labels)))
    # print ("AC  " + str(evaluation.matched_ac(pred, labels)))
    # print ("F1  " + str(evaluation.cal_F_score(pred, labels)[0]))

def node_classification_test(model):
    model.eval()
    z, _, _, _, _, _ = model(data.x, data.edge_index)
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    print('Accuracy of node classification is {}'.format(acc))
    return acc


print('Start training !!!')
stop_cnt = 0
best_idx = 0
patience = 200
min_loss = 1e9
real_epoch = 0
for epoch in range(1001):
    loss = train()
    if epoch % 20 == 0 and epoch > 0:
        print('epoch = {}'.format(epoch))
        tmp_mx_nmi,tmp_max_ac = test(model)
        max_nmi = max(max_nmi,tmp_mx_nmi)
        max_ac = max(max_ac,tmp_max_ac)
        node_classification_test(model)
    if loss < min_loss:
        min_loss = loss
        best_idx = epoch
        stop_cnt = 0
        torch.save(model.state_dict(), 'best_model.pkl')
    else:
        stop_cnt += 1
    if stop_cnt >= patience:
        real_epoch = epoch
        break

print('Loading {}th epoch'.format(best_idx))
model.load_state_dict(torch.load('best_model.pkl'))
print('Start testing !!!')
test(model)
node_classification_test(model)
print("max nmi&ac: ",max_nmi,max_ac)

#x1 = range(0, 1000, 20)
# x2 = range(0, 1000, 20)
# y1 = nmi_list[0:50]
# y2 = loss_list
# #y3 = DGI_loss_list
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, '.-')
# plt.title('ONLY DGI LOSS')
# plt.xlabel('Test NMI vs. epoches')
# plt.ylabel('Test NMI')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('DGI loss vs. epoches')
# plt.ylabel('DGI loss')
# # plt.subplot(3,1,3)
# # plt.plot(x3, y3, '.-')
# # plt.xlabel('DGI loss vs. epoches')
# # plt.ylabel('DGI loss')
# plt.show('DGI loss')
# plt.savefig("NMI_loss.jpg")


