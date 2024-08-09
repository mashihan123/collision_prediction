import os, random
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter, Sequential, Linear, BatchNorm1d
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, SGConv, global_add_pool, global_mean_pool


class Encoder(torch.nn.Module):
	"""
	A wrapper class for easier instantiation of pre-implemented graph encoders.
	Part of the code has been adapted from https://github.com/divelab/DIG.
	
	Args:
		feat_dim (int): The dimension of input node features.
		hidden_dim (int): The dimension of node-level (local) embeddings. 
		n_layers (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
		pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
			(default: :obj:`sum`)
		gnn (string, optional): The type of GNN layer, :obj:`gcn` or :obj:`gin` or :obj:`gat`
			or :obj:`graphsage` or :obj:`resgcn` or :obj:`sgc`. (default: :obj:`gcn`)
		bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
		node_level (bool, optional): If :obj:`True`, the encoder will output node level
			embedding (local representations). (default: :obj:`False`)
		graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
			embeddings (global representations). (default: :obj:`True`)
		edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
			compute the aggregation. (default: :obj:`False`)
			
	Note
	----
	For GCN and GIN encoders, the dimension of the output node-level (local) embedding will be 
	:obj:`hidden_dim`, whereas the node-level embedding will be :obj:`hidden_dim` * :obj:`n_layers`. 
	For ResGCN, the output embeddings for boths node and graphs will have dimensions :obj:`hidden_dim`.
			
	Examples
	--------
	>>> feat_dim = dataset[0].x.shape[1]
	>>> encoder = Encoder(feat_dim, 128, n_layers=3, gnn="gin")
	>>> encoder(some_batched_data).shape # graph-level embedding of shape [batch_size, 128*3]
	
	>>> encoder = Encoder(feat_dim, 128, n_layers=5, node_level=True, graph_level=False)
	>>> encoder(some_batched_data).shape # node-level embedding of shape [n_nodes, 128]
	
	>>> encoder = Encoder(feat_dim, 128, n_layers=5, node_level=True, graph_level=False)
	>>> encoder(some_batched_data) # a tuple of graph-level and node-level embeddings
	"""

	def __init__(self, feat_dim, hidden_dim, n_layers, gnn, pool="sum", bn=True, node_level=True, graph_level=True):
		super(Encoder, self).__init__()

		if gnn == "gcn":
			self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "gin":
			self.encoder = GIN(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "resgcn":
			self.encoder = ResGCN(feat_dim, hidden_dim, n_layers, pool)
		elif gnn == "gat":
			self.encoder = GAT(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "tgat":
			self.encoder = TemporalGAT(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "graphsage":
			self.encoder = GraphSAGE(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "sgc":
			self.encoder = SGC(feat_dim, hidden_dim, n_layers, pool, bn)

		self.node_level = node_level
		self.graph_level = graph_level

	def forward(self, data):
		z_g, z_n = self.encoder(data)
		if self.node_level and self.graph_level:
			return z_g, z_n
		elif self.graph_level:
			return z_g
		else:
			return z_n

	def save_checkpoint(self, save_path, optimizer, epoch, best_train_loss, best_val_loss, is_best):
		ckpt = {}
		ckpt["state"] = self.state_dict()
		ckpt["epoch"] = epoch
		ckpt["optimizer_state"] = optimizer.state_dict()
		ckpt["best_train_loss"] = best_train_loss
		ckpt["best_val_loss"] = best_val_loss
		torch.save(ckpt, os.path.join(save_path, "model.ckpt"))
		if is_best:
			torch.save(ckpt, os.path.join(save_path, "best_model.ckpt"))

	def load_checkpoint(self, load_path, optimizer):
		ckpt = torch.load(os.path.join(load_path, "best_model.ckpt"))
		self.load_state_dict(ckpt["state"])
		epoch = ckpt["epoch"]
		best_train_loss = ckpt["best_train_loss"]
		best_val_loss = ckpt["best_val_loss"]
		optimizer.load_state_dict(ckpt["optimizer_state"])
		return epoch, best_train_loss, best_val_loss



class GAT(torch.nn.Module):
    """
    Graph Attention Network from the paper `Graph Attention
    Networks <https://arxiv.org/abs/1710.10903>`.
    """

    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum",
                 heads=8, bn=False, xavier=False):
        super(GAT, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        a = torch.nn.ELU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GATConv(start_dim, hidden_dim, heads=heads, concat=False)
            if xavier:
                self.weights_init(conv)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, module):
        for m in module.modules():
            layers = []
            if isinstance(m, GATConv):
                layers = [m.lin_src, m.lin_dst]
            elif isinstance(m, Linear):
                layers = [m]
            
            for layer in layers:
                if hasattr(layer, 'weight'):
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)

    def forward(self, data):
		
        x, edge_index, batch, edge_attr = data

        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.acts[i](x)
            if hasattr(self, 'bns'):
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == "sum":
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep, x



class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        out = out[:, -1, :]  # Select the last element in the sequence for each batch
        
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_size=32, num_heads=4, dropout=0.1, forward_expansion=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, num_heads, dropout, forward_expansion)

    def create_mask(self, x):
        mask = torch.isnan(x).any(dim=-1)
        mask = ~mask  # Invert mask to match valid/invalid token representation
        return mask

    def forward(self, x):
        # Replace NaNs with zeros for embedding
        x = torch.nan_to_num(x, nan=0.0)

        embedded = self.embedding(x)
        normalized_embedded = self.layer_norm(embedded)
        mask = self.create_mask(x)
        out = self.transformer_block(normalized_embedded, normalized_embedded, normalized_embedded, mask)
        return out


class MaskedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MaskedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # Ensure the input is in float32
        x = x.float()
        
        # Create a mask to identify non-NaN elements
        mask = ~torch.isnan(x).any(dim=2)
        
        # Replace NaNs with zeros (or another appropriate value)
        x = torch.nan_to_num(x, nan=0.0)
        
        # Compute lengths and ensure no zero lengths
        lengths = mask.sum(dim=1)
        lengths[lengths == 0] = 1
        
        # Pack the sequence based on the mask
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through the LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)

        final_hidden_state = hn[-1]

        return final_hidden_state


class GraphClassificationModel(nn.Module):
	"""
	Model for graph classification.
	GNN Encoder followed by linear layer.
	
	Args:
		feat_dim (int): The dimension of input node features.
		hidden_dim (int): The dimension of node-level (local) embeddings. 
		n_layers (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
		gnn (string, optional): The type of GNN layer, :obj:`gcn` or :obj:`gin` or :obj:`gat`
			or :obj:`graphsage` or :obj:`resgcn` or :obj:`sgc`. (default: :obj:`gcn`)
		load (string, optional): The SSL model to be loaded. The GNN encoder will be
			initialized with pretrained SSL weights, and only the classifier head will
			be trained. Otherwise, GNN encoder and classifier head are trained end-to-end.
	"""

	def __init__(self, input_dim, hidden_dim, n_layers, gnn, output_dim, sequence_len, load=None):
		super(GraphClassificationModel, self).__init__()
		# self.transformer = TransformerModel(input_dim)
		self.sequence_len = sequence_len
		self.encoder = Encoder(input_dim, hidden_dim, n_layers=n_layers, gnn=gnn)

		if load:
			ckpt = torch.load(os.path.join("logs", load, "best_model.ckpt"))
			self.encoder.load_state_dict(ckpt["state"])
			for param in self.encoder.parameters():
				param.requires_grad = False

		if gnn in ["resgcn", "sgc"]:
			feat_dim = hidden_dim
		else:
			feat_dim = n_layers * hidden_dim

		self.lstm = MaskedLSTM(6, 6, 4)
		self.lstm_dim = hidden_dim*2+2
		self.classifier = nn.Sequential(
                            nn.Linear(feat_dim, 1000),
                            nn.ReLU(),
                            # nn.Linear(1000, 500),
                            # nn.Sigmoid(),
                            nn.Linear(1000, output_dim),
							# nn.Dropout(0.2),
                            # nn.Sigmoid()
                        	)
		# self.edge_regression = nn.Sequential(
		# 						nn.Linear(feat_dim, 1000),
        #                     	nn.ReLU(),
		# 						nn.Linear(1000, 500),
        #                     	nn.ReLU(),
		# 						nn.Dropout(0.2),
		# 						nn.Linear(500, 1)
		# 					)
		# self.edge_classifier = nn.Sequential(
		# 					nn.Linear(hidden_dim*2 + 2, 1000),
		# 					nn.ReLU(),
		# 					nn.Linear(1000, output_dim),
		# 					# nn.Sigmoid(),
		# 					)

	def forward(self, data):
		x_numeric, x_dummies, edge_index, batch, edge_attr = data
		lstm_output = self.lstm(x_numeric)
		x = torch.cat([lstm_output, x_dummies], 1)
		data1 = x, edge_index, batch, edge_attr
		embeddings = self.encoder(data1)
		# node_embeddings = embeddings[1]
		# x_src = node_embeddings[edge_index[0]]
		# x_dst = node_embeddings[edge_index[1]]
		# edge_feat = torch.cat([x_src, edge_attr, x_dst], dim=-1)
		graph_embeddings = embeddings[0]
		output = self.classifier(graph_embeddings)
		return output

	def save_checkpoint(self, save_path, optimizer, epoch, best_train_loss, best_val_loss, is_best):
		ckpt = {}
		ckpt["state"] = self.state_dict()
		ckpt["epoch"] = epoch
		ckpt["optimizer_state"] = optimizer.state_dict()
		ckpt["best_train_loss"] = best_train_loss
		ckpt["best_val_loss"] = best_val_loss
		torch.save(ckpt, os.path.join(save_path, "pred_model.ckpt"))
		if is_best:
			torch.save(ckpt, os.path.join(save_path, "best_pred_model.ckpt"))

	def load_checkpoint(self, load_path, optimizer):
		ckpt = torch.load(os.path.join(load_path, "best_pred_model.ckpt"))
		self.load_state_dict(ckpt["state"])
		epoch = ckpt["epoch"]
		best_train_loss = ckpt["best_train_loss"]
		best_val_loss = ckpt["best_val_loss"]
		optimizer.load_state_dict(ckpt["optimizer_state"])
		return epoch, best_train_loss, best_val_loss