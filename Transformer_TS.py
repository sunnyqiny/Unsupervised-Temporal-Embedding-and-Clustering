import numpy as np
import torch

import causal_convolution_layer
import Dataloader_ucr
import torch.nn.functional as F

from torch.utils.data import DataLoader
from kmeans_pytorch import kmeans

class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper

    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    """

    def __init__(self):
        super(TransformerTimeSeries, self).__init__()

        self.feature_size = 256
        self.dropout = 0.1
        self.num_layers = 3

        self.input_embedding = causal_convolution_layer.context_embedding(2, self.feature_size, 9)
        self.positional_embedding = torch.nn.Embedding(640, self.feature_size)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=8, dropout=self.dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=self.num_layers)

        self.fc1 = torch.nn.Linear(self.feature_size, 1)
        self.f_class = torch.nn.Linear(self.feature_size, 1)


    def forward(self, x, y, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)

        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)
        input_embedding = z_embedding + positional_embeddings
        transformer_embedding = self.transformer_encoder(input_embedding, attention_masks)

        transformer_reconstruction = self.transformer_decoder(transformer_embedding, attention_masks)
        output = self.fc1(transformer_reconstruction.permute(1, 0, 2))
        psu_class = self.f_class(transformer_reconstruction.permute(1, 0, 2))
        return output, psu_class, transformer_embedding, transformer_reconstruction


def loss_spectral(transformer_embedding, theta, pseudo_Y):
    # transformer_embedding: n * m; n - number of time-series; m - number of embedding size;
    # theta: similarity parameter；
    # pseudo_Y: dim： c * n; c - number of classes;

    ### Spectral Analysis ####
    ts = transformer_embedding
    mat_dist = euclidean_dist_mat(ts, ts)  # dim： n * n
    mat_G = mat_dist
    # mat_G = torch.exp((-1/theta**2)*torch.mul(mat_dist,mat_dist))
    mat_D = torch.diag(torch.sum(mat_G, 1, False))
    mat_L = mat_D - mat_G;
    loss_spect = torch.trace(torch.chain_matmul(pseudo_Y, mat_L, pseudo_Y.t()))

    print('loss_spectral: ', loss_spect)
    return loss_spect, mat_L

def loss_clustering(real_label, predict_label, transf_emb):
    real_label_uni = np.unique(real_label)
    loss_cluster = 0

    for label in real_label_uni:
        index = np.where(real_label==label)
        cluster = predict_label[index]
        center_real = np.mean(transf_emb[index, :], 1)

        index = np.asarray(index)

        cluster_uni = np.unique(cluster)
        for clu in cluster_uni:
            clu_index = np.where(cluster==clu)
            ori_index = index[0, clu_index]
            center_clu = np.mean(transf_emb[ori_index,:], 1)

            dist = F.pairwise_distance(torch.tensor(center_real), torch.tensor(center_clu), 2)
            loss_cluster = loss_cluster + dist

    print('loss_clustering: ', loss_cluster)
    return loss_cluster


def euclidean_dist_mat(x, y):
    x = x[0]
    y = y[0]

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = torch.clamp(dist, min=1e-12, max=1e12).sqrt()
    return dist

def Dp(y_pred,y_true,q):
    return max([q*(y_pred-y_true),(q-1)*(y_pred-y_true)])

def Rp_num_den(y_preds,y_trues,q):
    numerator = np.sum([Dp(y_pred,y_true,q) for y_pred,y_true in zip(y_preds,y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator,denominator


def train_epoch(model, train_dl, psu_Y_old, epoch, num_c):
    model.train()
    train_loss = 0
    n = 0
    lambda1 = 1
    lambda2 = 1

    print('lambda1, lambda2: ', lambda1, lambda2)

    for step, (x, y, attention_masks) in enumerate(train_dl):
        optimizer.zero_grad()
        y = torch.tensor(y, dtype=torch.float32)
        output, psu_class, transf_emb, transf_reconst = model(x.to(device), y.to(device), attention_masks[0].to(device))

        #update psu_Y_update with psu_class
        psu_Y_update = psu_class.squeeze()[:, 0:num_c]
        psu_Y_update = psu_Y_update.t()

        psu_Y_predict = psu_Y_update.argmax(dim=0)
        if epoch > 0:
            psu_Y_old = psu_Y_old.argmax(dim=0)

        t0 = np.floor(x.size(1) *0.9)
        t0 = t0.astype(int)

        loss_prediction = criterion(output.squeeze()[:, (t0 - 1):(x.size(1) - 1)], y.to(device)[:, t0:])  # not missing data
        loss_spectr, mat_L = loss_spectral(transf_emb, theta=1, pseudo_Y=psu_Y_update)

        emb_enc = transf_emb.permute(1,0,2)
        emb_enc = emb_enc.squeeze()[:,:,0]
        emb_enc = emb_enc.detach().numpy()

        loss_cluster = torch.log2(loss_clustering(psu_Y_old, psu_Y_predict, emb_enc))

        loss_prediction = torch.log2(loss_prediction)
        loss_spectr = torch.log2(loss_spectr)

        loss = loss_prediction + lambda2 * loss_spectr + lambda2 * loss_cluster
        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]

    # updating pseudo label matrix
    if epoch % 5 ==0 & epoch != 0:
        psu_Y_update = psu_Y_update - lambda1 * psu_Y_update * mat_L  # + (-lambda2/2 * 1/n *C)

    return train_loss / n, psu_Y_update, model, transf_emb, transf_reconst

def test_epoch(model, test_dl, num_c):
    with torch.no_grad():
        predictions = []
        observations = []

        for step, (x, y, attention_masks) in enumerate(test_dl):

            y = torch.tensor(y, dtype=torch.float32)

            output, psu_class, transf_emb, transf_reconst = model(x.to(device), y.to(device), attention_masks[0].to(device))

            psu_Y_class = psu_class.squeeze()[:, 0:num_c]
            psu_Y_class = psu_Y_class.t()

            t0 = np.floor(x.size(1) *0.9)
            t0 = t0.astype(int)

            for p, o in zip(output.squeeze()[:, (t0 - 1):(x.size(1) - 1)].cpu().numpy().tolist(),
                            y.to(device)[:, t0:].cpu().numpy().tolist()):  # not missing data
                # for p,o in zip(output.squeeze()[:,(t0-1-10):(t0+24-1-10)].cpu().numpy().tolist(),y.cuda()[:,(t0-10):].cpu().numpy().tolist()): # missing data
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den

    return Rp, output, psu_Y_class, transf_emb, transf_reconst

def get_attn(model, x, y, attention_masks):
    model.eval()
    with torch.no_grad():
        x = x.to(device);
        y = y.to(device);
        attention_masks = attention_masks.to(device);

        z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)
        z_embedding = model.input_embedding(z).permute(2, 0, 1)
        positional_embeddings = model.positional_embedding(x.type(torch.long)).permute(1, 0, 2)
        input_embedding = z_embedding + positional_embeddings

        attn_layer_i = []
        for layer in model.transformer_decoder.layers:
            attn_layer_i.append(
                layer.self_attn(input_embedding, input_embedding, input_embedding, attn_mask=attention_masks)[
                    -1].squeeze().cpu().detach().numpy())
            input_embedding = layer.forward(input_embedding, attention_masks)

        return attn_layer_i


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_dir = '/Users/qin/Desktop/ijcai_2021/UCR_TS_Archive_2015/'

    # file_dir = '../../root/UCR_TS_Archive_2015/'

    file = 'Beef'
    criterion = torch.nn.MSELoss()

    train = 1

    if train:
        train_filename = file_dir + file + '/' + file + '_TRAIN'
        test_filename = file_dir + file + '/' + file + '_TEST'

        print('\n===========================================================')
        print('data set: ', file)
        print('===========================================================')

        train_data = Dataloader_ucr.time_series_ucr(train_filename)
        test_data = Dataloader_ucr.time_series_ucr(test_filename)

        train_batch = train_data.fx.shape[0]
        valid_batch = train_data.fx.shape[0]
        test_batch = test_data.fx.shape[0]

        train_data.label, num_c = Dataloader_ucr.transfer_labels(train_data.label)
        test_data.label, num_ct = Dataloader_ucr.transfer_labels(test_data.label)

        train_dl = DataLoader(train_data,batch_size=train_batch,shuffle=False)
        test_dl = DataLoader(test_data,batch_size=test_batch)

        model = TransformerTimeSeries().to(device)

        lr = .0005 # learning rate

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epochs = 5

        # K-means for initialation
        x_train = torch.tensor(train_data.fx).float()
        psu_Y_update, cluster_centers = kmeans(
            X=x_train, num_clusters=num_c, distance='euclidean', device=torch.device(device)
        )

        train_epoch_loss = []
        test_epoch_loss = []
        Rp_best = 10

        ri_train_best = 0
        nmi_train_best = 0
        ri_test_best = 0
        nmi_test_best = 0
        epoch_best = 0


        for e, epoch in enumerate(range(epochs)):

            train_loss = []
            test_loss =[]

            l_t, psu_Y_update, model, transf_emb, transf_recons = train_epoch(model, train_dl, psu_Y_update, e, num_c)
            train_loss.append(l_t)

            psu_Y_train = psu_Y_update.argmax(dim=0)
            psu_Y_train = psu_Y_train.numpy()
            ri_train, nmi_train, acc, ari, anmi = Dataloader_ucr.evaluation(prediction=psu_Y_train, label=train_data.label)
            print('train: ri, nmi, acc, ari, anmi: ', ri_train, nmi_train, acc, ari, anmi)

            if ri_train > ri_train_best:
                ri_train_best = ri_train
            if nmi_train > nmi_train_best:
                nmi_train_best = nmi_train

            Rp, prediction, psu_Y_test, test_enc, test_dec = test_epoch(model, test_dl, num_c)

            psu_Y_test = psu_Y_test.argmax(dim=0)
            psu_Y_test = psu_Y_test.numpy()

            ri_test, nmi_test, acc, ari, anmi = Dataloader_ucr.evaluation(prediction=psu_Y_test, label=test_data.label)
            print('test: ri, nmi, acc, ari, anmi: ', ri_test, nmi_test, acc, ari, anmi)

            train_epoch_loss.append(np.mean(train_loss))

        print('train_epoch_loss: ', train_epoch_loss)
        np.savetxt('./embeddings/' + file + '_train_epoch_loss.txt', train_epoch_loss)





