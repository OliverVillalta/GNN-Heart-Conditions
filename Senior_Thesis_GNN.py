import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os 
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import classification_report
import ast

# Break data into a dictionary format (THIS WAS FOR WHEN I TRIED TO PAD ALL ENTRIES BUT KEPT GETTING ISSUES WITH COMPUTATIONAL POWER)
tmpdata_dict = {}
metadata = pd.read_csv('C:\\Users\\School Profile\\Documents\\Senior Project Thesis\\records\\prepared_code.csv')
length_list = (metadata['N'].unique()).tolist()
length_list.sort()
for n in length_list:
    tmpdata_dict[n] = []
    subset = metadata[metadata['N'] == n]
    for _, row in subset.iterrows():
        ecg_id = row['ECG_ID']
        reading_list = row['Reading']
        age = row['Age']
        sex = row['Sex']
        tmpdata_dict[n].append((ecg_id, reading_list, age, sex))

descriptions = ["Normal ECG", "Sinus tachycardia", "Sinus bradycardia", "Sinus arrhythmia", "Atrial premature complex(es)", "Atrial premature complexes, nonconducted", "Junctional premature complex(es)", "Junctional escape complex(es)", 
    "Atrial fibrillation", "Atrial flutter", "Junctional tachycardia", "Ventricular premature complex(es)", "Short PR interval", "AV conduction ratio N:D", "Prolonged PR interval", "Second-degree AV block, Mobitz type I (Wenckebach)", 
    "Second-degree AV block, Mobitz type II", "2:1 AV block", "AV block, varying conduction", "AV block, advanced (high-grade)", "AV block, complete (third-degree)", "Left anterior fascicular block", "Left posterior fascicular block", "Left bundle-branch block", 
    "Incomplete right bundle-branch block", "Right bundle-branch block", "Ventricular preexcitation", "Right-axis deviation", "Left-axis deviation", "Low voltage", "Left atrial enlargement", "Left ventricular hypertrophy", "Right ventricular hypertrophy", 
    "ST deviation", "ST deviation with T-wave change", "T-wave abnormality", "Prolonged QT interval", "TU fusion", "ST-T change due to ventricular hypertrophy", "Early repolarization", "Anterior MI", "Inferior MI", "Anteroseptal MI", "Extensive anterior MI", 
    "Occasional", "Frequent", "Acute", "Recent", "Old", "Couplets", "In a bigeminal pattern", "In a trigeminal pattern", "With a rapid ventricular response", "With a slow ventricular response", "With aberrancy", "Polymorphic", "Depression", "Elevation", "Inversion"]

desc_to_index = {desc: idx for idx, desc in enumerate(descriptions)}
num_classes = len(desc_to_index) 

def encode_reading_list(reading_list):
    vec = torch.zeros(num_classes)
    for desc in reading_list:
        desc = desc.strip()
        idx = desc_to_index.get(desc)
        if idx is not None:
            vec[idx] = 1
    return vec

top_freq = 10
time_features = 2
node_feature_dim = top_freq + time_features

# The amount of times I've had issues with this part is pretty astonishing... you'd think this would be easy...
def process_ecg_file(ecg_id, base_path, max_freq=40.0, sample_rate=500, top_k=top_freq, device='cuda'):
    file_path = os.path.join(base_path, f"{ecg_id}.h5")
    # print(f"Accessing file {file_path}")

    with h5py.File(file_path, "r") as f:
        ecg_data = torch.tensor(f['ecg'][:].T, dtype=torch.float32).to(device)  

    num_leads, signal_len = ecg_data.shape


    fft_vals = torch.fft.fft(ecg_data) 
    freqs = torch.fft.fftfreq(signal_len, d=1 / sample_rate).to(device)  


    freq_mask = (freqs < 0.5) | (freqs > max_freq)
    fft_vals[:, freq_mask] = 0


    fft_mag = torch.abs(fft_vals[:, :signal_len // 2])  


    available_k = min(top_k, fft_mag.shape[1])
    topk_vals = torch.topk(fft_mag, k=available_k, dim=1).values 

    if available_k < top_k:
        padding = torch.zeros((num_leads, top_k - available_k), device=device)
        topk_vals = torch.cat([topk_vals, padding], dim=1) 

    means = ecg_data.mean(dim=1, keepdim=True)  
    stds = ecg_data.std(dim=1, keepdim=True)    
    stats = torch.cat([means, stds], dim=1)     


    feature_tensor = torch.cat([topk_vals, stats], dim=1) 
    return feature_tensor

def truncate_signal(ecg_tensor, truncate_to=0.2):
    total_len = ecg_tensor.size(1)
    cutoff = int(total_len * truncate_to)
    return ecg_tensor[:, :cutoff]  


def encode_metadata_node(age, sex, dim):
    age_norm = min(float(age), 100.0) / 100.0
    sex_bin = 1.0 if str(sex) == 'F' else 0.0
    vec = [age_norm, sex_bin] + [0.0] * (dim - 2)
    return torch.tensor(vec, dtype=torch.float)


class ECGDataset(Dataset):
    def __init__(self, data_list, base_path, cache_dir="cached_ecg", metadata_dim=2, device="cuda"):
        super().__init__()
        self.data_list = data_list
        self.base_path = base_path
        self.cache_dir = cache_dir
        self.metadata_dim = metadata_dim
        self.device = device
        os.makedirs(self.cache_dir, exist_ok=True)

    def len(self):
        return len(self.data_list)

    @staticmethod
    def create_fully_connected_edge_index(num_nodes):
        row = []
        col = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    row.append(i)
                    col.append(j)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        return edge_index

    def get(self, idx):
        ecg_id, reading_str, age, sex = self.data_list[idx]
        reading_list = ast.literal_eval(reading_str)
        y = encode_reading_list(reading_list)
         

        feature_tensor = process_ecg_file(ecg_id, self.base_path, device=self.device)
        feature_tensor = feature_tensor.to(self.device)

        node_features = [feature_tensor[i] for i in range(12)]
        metadata_node_features = encode_metadata_node(age, sex, dim=node_feature_dim).to(self.device)
    
        node_features.append(metadata_node_features)

        x = torch.stack(node_features)
        edge_index = self.create_fully_connected_edge_index(x.size(0))
        return Data(x=x, y=y, edge_index=edge_index)


class DeepGATv2(nn.Module):
    def __init__(self, in_channels=node_feature_dim, hidden_channels=64, out_channels=num_classes, num_layers=3, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch) 
        return x

def train(model, loader, optimizer, device, scaler, class_weights):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()   

        loss_fn = nn.BCEWithLogitsLoss()
        

        encoded_labels = data.y.float().view(-1, num_classes) 
        
        if device.type == 'cuda':
            with autocast(device_type='cuda'):
                out = model(data.x, data.edge_index, data.batch)
        else:
            out = model(data.x, data.edge_index)


        assert out.shape == encoded_labels.shape, f"Output shape: {out.shape}, Label shape: {encoded_labels.shape}"
        
        loss = loss_fn(out, encoded_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()  
        
    return total_loss / len(loader)

# Useles...
def compute_class_weights(dataset, device='cpu'):
    total_samples = len(dataset)
    class_counts = torch.zeros(num_classes, device=device)
    

    for data in dataset:

        class_counts += data.y.sum(dim=0).to(device) 

    class_weights = total_samples / (num_classes * (class_counts + 1e-6))
    return class_weights

base_path = "C:\\Users\\School Profile\\Documents\\Senior Project Thesis\\records"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.2):
    model.eval()

    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        probs = torch.sigmoid(logits)

        preds = (probs > threshold).int().cpu().numpy()
        labels = data.y.view(-1, num_classes).int().cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=descriptions,
        zero_division=0
    ))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    print("Positive preds:", np.sum(all_preds))
    print("Positive labels:", np.sum(all_labels))
    return { "F1": f1, "Precision": precision, "Recall": recall, "Accuracy": accuracy}

f1_history = []
accuracy_history = []
loss_history = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_data = []
for batch in tmpdata_dict.values():
    all_data.extend(batch)

dataset = ECGDataset(all_data, base_path=base_path)
indices = list(range(len(dataset)))

train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_subset, batch_size=16, num_workers=0, pin_memory=False)
model = DeepGATv2(in_channels=node_feature_dim, hidden_channels=64, num_layers=3, heads=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
class_weights = compute_class_weights(dataset).to(device)

for epoch in range(1, 5):
    loss = train(model, train_loader, optimizer, device, scaler, class_weights)
    metrics = evaluate(model, val_loader, device)
    loss_history.append(loss)
    f1_history.append(metrics['F1'])
    accuracy_history.append(metrics['Accuracy'])

    print(f"[Epoch {epoch}] Train Loss: {loss:.4f} | Val F1: {metrics['F1']:.4f} | Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | Acc: {metrics['Accuracy']:.4f}")

epochs = range(1, len(loss_history) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(epochs, loss_history, label='Loss', color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs, f1_history, label='F1 Score', color='blue')
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title(f"F1 Score")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs, accuracy_history, label='Accuracy', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# I thought I wouldn't have to do this again
class CVDDataset(Dataset):
    def __init__(self, data_list, embed_dict, label_dict, metadata_dim=2, device='cuda'):
        super().__init__()
        self.data_list = data_list
        self.embeds = embed_dict
        self.labels = label_dict
        self.metadata_dim = metadata_dim
        self.device = device

    def len(self):
        return len(self.data_list)

    def edge_index_fc(self, num_nodes):
        row, col = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    row.append(i)
                    col.append(j)
        return torch.tensor([row, col], dtype=torch.long)

    def get(self, idx):
        ecg_id, _, age, sex = self.data_list[idx]
        y = torch.tensor([self.labels.get(ecg_id, 0)], dtype=torch.float)
        x = self.embeds[ecg_id].to(self.device)
        ei = self.edge_index_fc(x.size(0))
        return Data(x=x, y=y, edge_index=ei)


class CVDGCN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, ei, batch):
        x = self.conv1(x, ei)
        x = F.relu(x)
        x = self.conv2(x, ei)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x).view(-1)


def train_bin(model, loader, opt, device):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = crit(out, data.y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

# Just so my computer doesn't die
@torch.no_grad()
def eval_bin(model, loader, device):
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = (torch.sigmoid(out) > 0.5).int().cpu().numpy()
        lbl = data.y.int().cpu().numpy()
        preds.append(pred)
        labels.append(lbl)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return f1, prec, rec, acc


def make_bin_labels(data_dict, mode='abnormal'):
    labels = {}
    for batch in data_dict.values():
        for ecg_id, reading_str, age, sex in batch:
            readings = ast.literal_eval(reading_str)
            if mode == 'abnormal':
                label = int(any(r != "Normal ECG" for r in readings))
            elif mode == 'mi':
                label = int(any("MI" in r for r in readings))
            else:
                label = 0
            labels[ecg_id] = label
    return labels


model.eval()
embed_dict = {}
with torch.no_grad():
    for i in range(len(dataset)):
        data = dataset.get(i).to(device)
        x = model.convs[0](data.x, data.edge_index)
        for conv in model.convs[1:-1]:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=False)
        x = model.convs[-1](x, data.edge_index)
        embed_dict[all_data[i][0]] = x.cpu()

label_dict = make_bin_labels(tmpdata_dict, mode='abnormal')
cvd_dataset = CVDDataset(all_data, embed_dict, label_dict, device=device)
train_idx, val_idx = train_test_split(list(range(len(cvd_dataset))), test_size=0.2, random_state=42)
cvd_train = torch.utils.data.Subset(cvd_dataset, train_idx)
cvd_val = torch.utils.data.Subset(cvd_dataset, val_idx)
train_loader = DataLoader(cvd_train, batch_size=16, shuffle=True)
val_loader = DataLoader(cvd_val, batch_size=16)
gcn_model = CVDGCN(in_channels=embed_dict[next(iter(embed_dict))].shape[1]).to(device)
opt = torch.optim.Adam(gcn_model.parameters(), lr=1e-3)

gcn_f1, gcn_acc = [], []
for epoch in range(1, 6):
    loss = train_bin(gcn_model, train_loader, opt, device)
    f1, prec, rec, acc = eval_bin(gcn_model, val_loader, device)
    gcn_f1.append(f1)
    gcn_acc.append(acc)
    print(f"[GCN Epoch {epoch}] Loss: {loss:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Acc: {acc:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 6), gcn_f1, marker='o')
plt.title("GCN F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, 6), gcn_acc, marker='o', color='green')
plt.title("GCN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()
