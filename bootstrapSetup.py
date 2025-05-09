import torch
import torch.nn as nn
from CircuitSimulator import *
import numpy as np
from diffusers import UNet2DConditionModel
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Dataset
from matplotlib.colors import ListedColormap
import hashlib
from itertools import combinations
from tqdm import tqdm
import random
from collections import defaultdict
import os
import pickle

# Initial circuits

InitialCicuits = torch.tensor([
    [
        [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 3, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    ],
])

InitialTables = torch.tensor([
    [
        [0, 0, 2, 2, 0, 2, 2, 2 ],
        [1, 0, 2, 2, 0, 2, 2, 2 ],
        [0, 1, 2, 2, 0, 2, 2, 2 ],
        [1, 1, 2, 2, 1, 2, 2, 2 ],
    ],
    [
        [0, 0, 2, 2, 2, 0, 2, 2 ],
        [1, 0, 2, 2, 2, 0, 2, 2 ],
        [0, 1, 2, 2, 2, 0, 2, 2 ],
        [1, 1, 2, 2, 2, 1, 2, 2 ],
    ],
    [
        [0, 0, 2, 2, 2, 2, 0, 2 ],
        [1, 0, 2, 2, 2, 2, 0, 2 ],
        [0, 1, 2, 2, 2, 2, 0, 2 ],
        [1, 1, 2, 2, 2, 2, 1, 2 ],
    ],
    [
        [0, 0, 2, 2, 2, 2, 2, 0 ],
        [1, 0, 2, 2, 2, 2, 2, 0 ],
        [0, 1, 2, 2, 2, 2, 2, 0 ],
        [1, 1, 2, 2, 2, 2, 2, 1 ],
    ],
])

# Transform initial data into data for neural network model
# Also has a few utility functions for data manipulation and visualization

def imageToProbabilities(image, numCategories):
    # The categorical scheduler expects each channel to describe the probability of a pixel being of that class
    # Therefore, a RawCircuit, with one channel, needs to be expanded to have numCategories channels

    bs, h, w = image.shape
    
    imageProbabilites = torch.zeros(bs, numCategories, h, w)
    for b in range(bs):
        for i in range(h):
            for j in range(w):
                pixelClass = image[b, i, j]
                imageProbabilites[b, pixelClass, i, j] = 1.0

    return imageProbabilites

def tablePadding(truthTable):
    #Takes a truthtable and adds rows to fix row amount to 16.
    rows = truthTable.shape[0]

    padding = torch.full((16 - rows, truthTable.shape[1]), 2.0)
    return torch.cat((truthTable, padding), dim=0).long()

def countWires(circuit):
    sum = 0
    for i in range(circuit.shape[0]):
        for j in range(circuit.shape[1]):
            if circuit[i][j] >= 1:
                sum += min(2, circuit[i][j])
    return sum

def printCircuit(circuit):
    cmap = ListedColormap(['white', 'black', 'red', 'blue'])
    fig, ax = plt.subplots(figsize=(2, 2))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(circuit, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    plt.tight_layout()
    plt.show()

# Function to get the hash of a tensor
def get_tensor_hash(tensor):
    # Flatten the tensor and convert to tuple (hashable)
    tensor_tuple = tuple(tensor.flatten().tolist())
    
    # Compute the hash using hashlib (SHA-256)
    hash_object = hashlib.sha256(str(tensor_tuple).encode())
    return hash_object.hexdigest()

# lists to dictionary
def CreateTrainingDictionary(inpDict, rawCircuits, tables):
    with torch.no_grad():
        shorterCircuits = 0
        for table, circuit in zip(tables, rawCircuits):
            truth = tablePadding(table)
            key = truth.numpy().tobytes() #Hashable
            wiresTotal = countWires(circuit)
            
            if key not in inpDict:
                inpDict[key] = (imageToProbabilities(circuit.unsqueeze(0), 4).squeeze(), wiresTotal)
            else: 
                oldCircuit, wireAmount = inpDict[key]
                if wireAmount > wiresTotal: # If new has less wires
                    inpDict[key] = (imageToProbabilities(circuit.unsqueeze(0), 4).squeeze(), wiresTotal)
                    shorterCircuits += 1

        return inpDict, shorterCircuits

class CircuitDataset(Dataset):
    def __init__(self, inpDict):
        self.dict = inpDict
        self.keys = list(self.dict.keys())

        self.key_tensors = [
            torch.from_numpy(np.frombuffer(k).copy()).long().reshape(16, 8)
            for k in self.keys
        ]

        self.hashes = [get_tensor_hash(torch.argmax(circuit, dim=0, keepdim=True))
                       for circuit, _ in list(inpDict.values())]

    def __len__(self):
        return len(self.dict)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        circuit, wireAmount = self.dict[key]
        
        return circuit, self.key_tensors[idx]

# Mostly a copy of CircuitDataset, but with the item count inflated, to match previous tests.
class LoopingDataset(Dataset):
    def __init__(self, inpDict):
        self.dict = inpDict
        self.keys = list(self.dict.keys())

        self.key_tensors = [
            torch.from_numpy(np.frombuffer(k).copy()).long().reshape(16, 8)
            for k in self.keys
        ]

        self.hashes = [get_tensor_hash(torch.argmax(circuit, dim=0, keepdim=True))
                       for circuit, _ in list(inpDict.values())]
    def __len__(self):
        normalLen = len(self.dict) # Is 4
        return normalLen * 4 # previous tests had 16 
    
    def __getitem__(self, idx):
        key = self.keys[idx % len(self.dict)]
        circuit, wireAmount = self.dict[key]
        
        return circuit, self.key_tensors[idx % len(self.dict)]


trainingDict, _ = CreateTrainingDictionary({}, InitialCicuits, InitialTables)
dataset = LoopingDataset(trainingDict)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

print("dataset generated as 'loader")

# Set truthtable embedder

class RowEmbedder(nn.Module):
    def __init__(self, num_categories, vector_length, embedding_dim):
        super().__init__()
        self.shared_embed = nn.Embedding(num_categories, embedding_dim)
        self.position_weights = nn.Parameter(torch.ones(vector_length, embedding_dim))
        self.position_bias = nn.Parameter(torch.zeros(vector_length, embedding_dim))
        
    def forward(self, x):
        shared = self.shared_embed(x)  
        # Apply position-specific scaling and shifting
        return shared * self.position_weights + self.position_bias
    

class TabularTransformer(nn.Module):
    def __init__(self, num_categories, num_features, d_model):
        super().__init__()
        self.d_model = d_model

        self.row_embedding = RowEmbedder(num_categories, num_features, d_model) #num_categories, vector_length, embedding_dim

        for param in self.row_embedding.parameters():
            param.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * num_features,
            nhead=8,
            dim_feedforward=2*d_model * num_features,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6,
        )

        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        #batch_size, num_rows, num_cols = x.shape

        rows = self.row_embedding(x) #bs, rows, columns, embedding
        #rows = rows.mean(dim=2)
        rows = rows.flatten(start_dim=2) 
        
    
        transformed = self.transformer(rows)

        return transformed

transformer = TabularTransformer(3, 8, 16)
transformer.load_state_dict(torch.load('data/tabular_transformer.pt'))
transformer.eval()



print("embedder created as transformer")
print(f"Embedded shape: {transformer(InitialTables)[0].shape}")


# Set up scheduler

class CategoricalScheduler:
    def __init__(self, TrainSteps = 200, numCategories = 4, betaStart = 0.0001, betaEnd = 0.02):
        self.TrainSteps = TrainSteps
        self.noiseDevice = 'cpu'
        self.numCategories = numCategories

        self.betas = torch.linspace(betaStart, betaEnd, TrainSteps, device=self.noiseDevice)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def addNoise(self, imageBatch, time):
        bs, ch, w, h = imageBatch.shape

        with torch.no_grad():
            alpha_t = self.alpha_cumprod[time].view(-1, 1, 1, 1) # Translates shape (1,) -> (1, 1, 1, 1)

            # the per pixel probability distribution of the categories
            currentProbabilities = imageBatch

            # The chance of each state per pixel when noised            
            updatedProbabilities = currentProbabilities * alpha_t + (1 - alpha_t) / self.numCategories 
            updatedProbabilities = updatedProbabilities.permute(0, 2, 3, 1) # reshape such that it is flattened correctly below
            updatedProbabilities = updatedProbabilities.reshape(bs*w*h, self.numCategories)  
            

            # 1 Sample per value
            categoricalNoise = torch.multinomial(updatedProbabilities, 1, replacement=True)
            categoricalNoise = categoricalNoise.view(bs, w, h) # Shape: [bs, w, h]

            noisedImages = F.one_hot(categoricalNoise, num_classes=self.numCategories)
            noisedImages = noisedImages.permute(0, 3, 1, 2) # [bs, num_classes, w, h]

            return noisedImages

scheduler = CategoricalScheduler()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Create Diffuser

class CategoricalDiffusionModel(nn.Module):
    def __init__(self, imageSize, numCategories, embeddingSize, attentionHeads=8, guidanceProb=0.1):
        super().__init__()
        self.guidance_prob=guidanceProb
        self.model = UNet2DConditionModel(
            sample_size=imageSize, 
            in_channels=numCategories,  # Image channels
            out_channels=numCategories,
            cross_attention_dim=embeddingSize,  # Matches mbedding's token dim 
            attention_head_dim=attentionHeads,     # Smaller head dim for efficiency
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 128), 
            down_block_types=(
                "CrossAttnDownBlock2D",  
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
        )
        
    def forward(self, x, t, embedding):
        bs, ch, w, h = x.shape

        #Use guidance probability, to help the model learn that 'circuit behvaior' 
        # is a general feature and not specific to a particular embedding.
        if self.training:
            mask = torch.rand_like(embedding, device=x.device) < self.guidance_prob
            embedding[mask] = 0

        return self.model(x, t, encoder_hidden_states=embedding).sample
    

# Sockets used for simulating
inpSockets = [
    (Socket("inp0", True), (-1, 0)), 
    (Socket("inp1", True), (-1, 4)), 
    (Socket("inp2", True), (-1, 8)), 
    (Socket("inp3", True), (-1, 12)), 
]

outSockets = [  
    (Socket("out0", False), (13, 0)), 
    (Socket("out1", False), (13, 4)), 
    (Socket("out2", False), (13, 8)), 
    (Socket("out3", False), (13, 12)),
]   


def GetTruthTable(inputIndexes, outputIndexes, socketMap, connectionMap, orders):
    # Orders is a list of the order in which input sockets are activated.
    # # Truthtables usually do not make use of activation order, but the simulation needs it.

    truthTable = torch.ones(size=(2**len(inputIndexes), 8)) * 2

    row = 0
    #try
    for order in orders:
        for socket in socketMap.keys():
            socket.state = False
        
        # Gives the state of sockets at the end. But only output sockets are relevant, 
        # and those are available through outSockets[i][0]
        _ = Simulate(connectionMap, socketMap, order) 

        for index in inputIndexes:
            truthTable[row][index] = 0

        for socket in order:
            prefix, idnum, comptype = socket.name
            truthTable[row][int(idnum)] = 1

        for i in outputIndexes:
            truthTable[row][i + 4] = outSockets[i][0].state

        row += 1

    return truthTable