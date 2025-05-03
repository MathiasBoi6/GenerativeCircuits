from bootstrapSetup import *
from datetime import datetime


EXPERIMENT_NAME= "SquareReqRemoved"




# Training Function

def trainCatModel(catModel, dataloader, embeddingModel, losses, stopAccuracy = 0.05, batchLimit = 100, PLOTUPDATES = True):
    catModel.train()
    updateRate = 10 # For plotting purposes

    loss_fn = nn.CrossEntropyLoss() #nn.MSELoss(reduction='sum')

    if PLOTUPDATES:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    opt = torch.optim.Adam(catModel.parameters(), lr=1e-4) 


    batchNum = 0
    isTraining = True
    while isTraining:
        for batch, labels in dataloader:

            timesteps = torch.randint(0, scheduler.TrainSteps - 1, (batch.shape[0],), device='cpu').long()

            noisyImgs = scheduler.addNoise(batch, timesteps)

            embeddings = embeddingModel(labels)

            pred = catModel(noisyImgs.to(device).float(), timesteps.to(device), embeddings.to(device))

            loss = loss_fn(pred, batch.to(device).argmax(axis=1)) 

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if PLOTUPDATES:
                if len(losses) % updateRate == 0:
                    clear_output(wait=True)  
                    ax.clear()
                    plt.plot(losses)
                    plt.xlabel('Batch')
                    display(fig)
                    
            batchNum += 1
            if loss < stopAccuracy:
                    isTraining = False
                    break
            elif batchNum >= batchLimit:
                isTraining = False
                break
    
    if PLOTUPDATES:
        clear_output(wait=True)  
        ax.clear()
    plt.plot(losses)
    plt.xlabel('Batch')
    if PLOTUPDATES:
        display(fig)

    return catModel, losses


def prePruning(connectionMap, socketMap):
    prunedConnections = []
    for connection in connectionMap:
        relevantConnection = any(
            component.isSource for component in connectionMap[connection]
        ) and any(
            not component.isSource for component in connectionMap[connection]
        )
        
        if not relevantConnection:
            prunedConnections.append(connection)

    for connection in prunedConnections:
        del connectionMap[connection]  

    connectedInputs = []
    connectedOutputs = []
    for socket, connections in socketMap.items():
        for connection in list(connections):
            if connection in prunedConnections:
                socketMap[socket].remove(connection)
                
        prefix, compid, comptype = socket.name
        
        if prefix == "inp" and bool(socketMap[socket]):
            connectedInputs.append(int(compid))
        elif prefix == "out" and bool(socketMap[socket]):
            connectedOutputs.append(int(compid))
    
    return connectedInputs, connectedOutputs



def generateSocketCombinations(inputSockets, relevantIndexes):
    #inputSockets = All available input sockets
    #relevantIndexes = the indexes of sockets that are connected to components
    
    usedSockets = [inputSockets[i][0] for i in relevantIndexes]
    
    # Generate all combinations (power set)
    orderList = []
    for r in range(len(usedSockets) + 1):
        for combo in combinations(usedSockets, r):
            orderList.append(list(combo))
    
    return orderList


def afterPruning(circuit, wiresets, connectionMap):
    # Prunes wires, from pruned connections
    # And replaces gates that aren't connected on all inputs and outputs with a wire.
    circuitWasUpdated = False

    for i in range(circuit.shape[0]):
        for j in range(circuit.shape[1]):
            if wiresets[i][j] not in connectionMap and circuit[i][j] == 1:
                circuit[i][j] = 0
                circuitWasUpdated = True

            elif circuit[i][j] > 1: # If gate
                #Check if it should be replaced with a wire

                try:
                    if (circuit[i][j + 1] == 0 or # If emmitter is 0
                        circuit[i][j - 1] == 0): # If base is 0

                        circuit[i][j] = 1 
                        circuitWasUpdated = True
                except: #
                    circuit[i][j] = 1 
                    circuitWasUpdated = True

                if (
                    (i < circuit.shape[1] - 1 and circuit[i + 1][j] == 0) 
                    and
                    (i - 1 < 0 and circuit[i - 1][j] == 0)):
                    circuit[i][j] = 1 
                    circuitWasUpdated = True                

    #Because there gate positions aren't stored, socket map will have to be recreated, but then connetionmap will also need to be updated.
    return circuit, connectionMap, circuitWasUpdated

# Prunes excess wires from the circuit.
def pruneExcessWires(circuit):
    circuitDirty = False
    
    for j in range(1, circuit.shape[0] - 1):
        for i in range(circuit.shape[1]):
            center = circuit[i][j]
            up     = circuit[i - 1][j] if i > 0 else 0
            down   = circuit[i + 1][j] if i < circuit.shape[0] - 1 else 0
            left   = circuit[i][j - 1] 
            right  = circuit[i][j + 1] 

            crossValue = center + up + down + left + right

            if crossValue < 3 and circuit[i][j] == 1:
                circuit[i][j] = 0
                circuitDirty = True
    
    if circuitDirty:
        return pruneExcessWires(circuit)
    else: return circuit

# Generate N Circuits
def GenerateN(model, amount = 100):
    model.eval()
    
    ## GENERATE CIRCUITS
    with torch.no_grad():
        batchX = torch.randint(size=(amount, 13, 13), high = 3, low = 0, device=device )
        batchX = imageToProbabilities(batchX, 4)

        batchY = torch.randint(size=(amount, 16, 8), high = 2, low = 0, device='cpu' )
        batchY = transformer(batchY).to(device)
        
        # Inference with trainingsteps // 64 + 1, for faster generations. Even with this few steps, generations become perfect after training, due to small dataset.
        stepDivisor = 64
        inferenceSteps = torch.linspace(scheduler.TrainSteps-1, 1, scheduler.TrainSteps // stepDivisor, device='cpu').long()

        
        for t in inferenceSteps:
            residual_cond = model(batchX.to(device).float(), t, batchY.float()) 
            residual = F.softmax(residual_cond, dim=1)
            batchX = scheduler.addNoise(residual.to('cpu'), t - 1).float()
        batchX = F.softmax(model(batchX.to(device), 0, batchY), dim=1)

        # Get circuit values
        argmaxedBatch = torch.argmax(batchX, dim=1, keepdim=True).cpu().numpy().squeeze()

        return argmaxedBatch
    

def requireGate(circuit):
    rows = circuit.shape[0]
    cols = circuit.shape[1]

    for i in range(rows):
        for j in range(cols):
            if circuit[i][j] > 1:
                return True
    return False

def getTables(circuits):
    #Prunes
    #Checks hashes
    #Simulates
    #Then, generates tables
    with torch.no_grad():
        tables = []
        simulatedCircuits = []
        simulatedCircuitHashes = []

        unique_hashes = {}
        circuitsErrors = defaultdict(int)
        for i in range((circuits.shape[0])):
            circuitDirty = True # Pruning flag

            while circuitDirty:
                socketMap, wireSets = GetSocketMap(circuits[i], inpSockets + outSockets)
                connectionMap = GetConnectionMap(socketMap)


                inps, outs = prePruning(connectionMap, socketMap)
                circuit, connectionMap, circuitDirty = afterPruning(circuits[i], wireSets, connectionMap)
                circuit = pruneExcessWires(circuit)
            
            hash = get_tensor_hash(circuit)
            if hash not in unique_hashes:
                if hash not in dataset.hashes:

                    #Get orderlist
                    if len(inps) > 1 and len(outs) > 0: # Bias to have only circuits with at least 2 inputs
                        orderlist = generateSocketCombinations(inpSockets, inps)

                        try:    
                            if requireGate(circuit):
                                truthTable = GetTruthTable(inps, outs, socketMap, connectionMap, orderlist)
                                tables.append(truthTable)
                                simulatedCircuits.append(circuit)
                                simulatedCircuitHashes.append(hash)
                                unique_hashes[hash] = i
                            else:
                                circuitsErrors["No gates"] += 1        

                        except Exception as e:
                            circuitsErrors[str(e)] += 1
                    else:
                        circuitsErrors["Not Sufficiently Connected"] += 1
                else:
                    circuitsErrors["Duplicate"] += 1
            else:
                circuitsErrors["New Duplicate"] += 1
                
        return tables, simulatedCircuits, simulatedCircuitHashes, circuitsErrors
    

# GenerateN and getTables combined.
def GenerateAndSimulate(model, amount = 100): 
    argmaxCircuits = GenerateN(model, amount)
    tables, simulatedCircuits, simulatedCircuitsHashes, errors = getTables(argmaxCircuits)

    return tables, simulatedCircuits, simulatedCircuitsHashes, errors


def SampleCircuits(model, iterations = 10, samples = 512):
    newCircuits = []
    newCircuitHashes = []
    newTables = []

    errors = defaultdict(int)
    for i in range(iterations):
        t, a, h, e = GenerateAndSimulate(model, amount=samples)
        for circuit, hash, table in zip(a, h, t):
            if hash not in newCircuitHashes:
                newCircuitHashes.append(hash)   
                newCircuits.append(circuit)
                newTables.append(table)

        for error, count in e.items():
            errors[error] += count

    return newCircuits, newTables, errors


def save_progress(model, circuitErrors, trainingDict, trainingDictSize, dataIncrease, amountShorter, amountCircuitsGenerated):
    # Get today's date in YYYY-MM-DD format
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # Construct the result directory path
    result_dir = os.path.join('output', today_str)
    os.makedirs(result_dir, exist_ok=True)

    # Define filenames with 'experiment_' prefix, no date
    model_filename = f'{EXPERIMENT_NAME}_model.pt'
    data_filename = f'{EXPERIMENT_NAME}_training_data.pkl'

    # Save model
    torch.save(model.state_dict(), os.path.join(result_dir, model_filename))

    # Save other data
    save_data = {
        'circuitErrors': circuitErrors,
        'trainingDict': trainingDict,
        'trainingDictSize': trainingDictSize,
        'dataIncrease': dataIncrease,
        'amountCircuitsGenerated': amountCircuitsGenerated,
        'amountShorter': amountShorter
    }

    with open(os.path.join(result_dir, data_filename), 'wb') as f:
        pickle.dump(save_data, f)



def Bootstrap(epochs, model, loss, trainingDict): 
    dataset = CircuitDataset(trainingDict)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    errorClasses = ["No gates", "Not Sufficiently Connected", "Duplicate", "New Duplicate"]
    circuitErrors = {error: [] for error in errorClasses}

    trainingDictSize = [len(trainingDict)]
    dataIncrease = []
    amountCircuitsGenerated = []
    amountShorter = []

    for epoch in tqdm(range(epochs)):
        #Train model
        model, loss = trainCatModel(model, loader, transformer, loss, PLOTUPDATES=False)

        # Generate new data
        newCircuits, newTables, errors = SampleCircuits(model, samples=256)
        for error in errorClasses:
            circuitErrors[error].append(errors[error])
        amountCircuitsGenerated.append(len(newCircuits))
            
        # Update training dictionary
        pastLen = len(trainingDict)
        trainingDict, fasterCircuits = CreateTrainingDictionary(trainingDict, torch.tensor(np.array(newCircuits)), newTables)
        dataset = CircuitDataset(trainingDict)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        dataIncrease.append(len(trainingDict) - pastLen)
        trainingDictSize.append(len(trainingDict))
        amountShorter.append(fasterCircuits)

        # Save progress
        save_progress(model, circuitErrors, trainingDict, trainingDictSize, dataIncrease, amountShorter, amountCircuitsGenerated)


    return model, loss, trainingDict


model = CategoricalDiffusionModel(13, 4, 128).to(device)
model, loss = trainCatModel(model, loader, transformer, [], PLOTUPDATES=False)
model, loss, trainingDict = Bootstrap(2, model, loss,trainingDict)
