import string
import re
import numpy as np
from enum import Enum, auto

"""
RawCircuit -> WireSetGrid
RawCircuit -> list of (Gate + position)
		        ^Gates

WireSetGrid + SocketList -> SocketConnections 
WireSetGrid + Gates -> GateSocketConnections 
GateSocketConnections + SocketConnections -> SocketMap

SocketMap -> ConnectionMap
"""


# Recognized components in circuit image / RawCircuit
class ComponentType(Enum):
  nothing = 0
  wire = auto()
  AND = auto()
  NAND = auto()

def doesGateFire(base, collector, prefix):
    if prefix == "AND":
        return base.state * collector.state
    elif prefix == "NAND":
        return not (base.state and collector.state)
    else:
        raise ValueError(f"Unknown component type: {prefix}")

RawCircuit = np.ndarray

WireSet = int # An ID for connected wires

# Inp, Out, and Gate pins
# Not used much. 
class Socket:
    def __init__(self, id : string, isSource : bool,  state = False):
        self.state = state
        self.id = id
        self.isSource = isSource # Whether this socket powers connected wires

    def __repr__(self):
        return f"{self.id}"
    
    @property
    def name(self):
        match = re.match(r"([a-zA-Z]+)(\d*)([a-zA-Z]*)", self.id)
        if match:
            # prefix, component id, component type
            # Example, 'transistor0emitter'
            return match.groups() 
        raise NameError(f"Name of socket not allowed: {self.id}") # This should be checked initialy instead
    
    def __eq__(self, other):
        if isinstance(other, Socket):
            return self.id == other
        elif isinstance(other, str):  # Compare directly with strings
            return self.id == other
        return False

    def __hash__(self):
        return hash(self.id)


# When quereing an array index outside of bounds, returns default value.
class SafeGrid:
    def __init__(self, grid, default=False):
        self.grid = grid
        self.default = default

    def __getitem__(self, index):
        x, y = index
        if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
            return self.grid[x][y]
        return self.default
    

# Local function
# RawCircuit -> WireSetGrid
def WireSets(WireImage : RawCircuit):
    connections = np.full_like(WireImage, None, dtype=object) 
    WireImage = np.pad(WireImage, pad_width=((1, 0), (1, 0)), mode='constant') # This is only padded to avoid index out of bounds errors.

    def TraceSetEnd(position, newParent):
        # Traces the connections until it reaches an element pointing to itself (the end)
        # Each value iterated over in the process have their parent set to the end value.
        
        while connections[position] is not newParent:
            nextPosition = connections[position]
            connections[position] = newParent
            position = nextPosition

    # Create disjoint sets of connected wires
    for i in range(connections.shape[0]):
        for j in range(connections.shape[1]):
            if WireImage[i + 1, j + 1] == 1: # If current pixel has a wire
                if WireImage[i, j + 1] >= 1: # and the pixel above has a wire
                    connections[i, j] = connections[i - 1, j]

                    if WireImage[i + 1, j] == 1: # if both left and up have a wire, set up to use above as parent.
                        TraceSetEnd((i, j - 1), connections[i, j])

                elif WireImage[i + 1, j] == 1: # or the pixel to the left has a wire
                    connections[i, j] = connections[i, j - 1]
                else: # If no connections found, set as set containing self
                    connections[i, j] = (i, j)

            # If current pixel is a gate, connect top to bottom
            elif WireImage[i + 1, j + 1] > 1: 
                if WireImage[i, j + 1] >= 1:
                    connections[i, j] = connections[i - 1, j]
                else: # If no connections found, set as set containing self
                    connections[i, j] = (i, j)

    def TraceSet(position):
        # This should be changed to set values during travel as a dynamic programming approach.
        while position is not connections[position]:
            position = connections[position]
        
        return position

    connectionDict = {}
    setCount = 0
    WireSetGrid = np.zeros_like(connections)
    for i in range(connections.shape[0]):
        for j in range(connections.shape[1]):
            if connections[i, j] is not None:
                setRepressentative = TraceSet(connections[i, j])

                if setRepressentative not in connectionDict:
                    setCount += 1
                    connectionDict[setRepressentative] = setCount

                WireSetGrid[i, j] = connectionDict[setRepressentative]

    return WireSetGrid, setCount


# RawCircuit + SocketList -> SocketMap
# Through
# WireSetGrid + SocketList -> SocketConnections 
# WireSetGrid + Gates -> GateSocketConnections 
# GateSocketConnections + SocketConnections -> SocketMap
def GetSocketMap(circuit : RawCircuit, SocketList : list[Socket]):
    WireSetGrid, setCount = WireSets(circuit)

    Gates = []
    for i in range(circuit.shape[0]):
        for j in range(circuit.shape[1]):
            if circuit[i][j] > 1:
                Gates.append((ComponentType(circuit[i][j]).name, (i,j)))

    safeGrid = SafeGrid(WireSetGrid)
    ComponentMap = {}

    # From a point in safeGrid, get connected wireSets
    def pointConnections(point):
        x, y = point

        # down, up, left, right
        potentialConnections = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
        connections = [safeGrid[yy, xx] 
                       if safeGrid[yy, xx] else None 
                       for yy, xx in potentialConnections]
        return connections 
    
    setComprehendConncetions = lambda set: {value for value in set if value is not None}

    # Get wireSets connected to each socket
    for socket, position in SocketList:
        connectionValues = pointConnections(position)
        ComponentMap[socket] = setComprehendConncetions(connectionValues)

    # Get wireSets connected to each gate
    gateId = 0
    for gate, position in Gates:
        base = set() # Left pin, input
        collector = set() # Vertical pin, input
        emitter = set() # Right pin, output
        y, x = position
        down, up, left, right = pointConnections((x, y))

        base = setComprehendConncetions([left])
        collector = setComprehendConncetions([up, down])
        emitter = setComprehendConncetions([right])

        ComponentMap[Socket(f"{gate}{gateId}base", False)] = base
        ComponentMap[Socket(f"{gate}{gateId}collector", False)] = collector
        ComponentMap[Socket(f"{gate}{gateId}emitter", True)] = emitter
        
        gateId += 1

    return ComponentMap

# Map from wireSets to components
def GetConnectionMap(ComponentMap : dict):
    ConnectionMap = {}

    def addToMap(key, name):
        if key not in ConnectionMap:
            ConnectionMap[key] = []
        ConnectionMap[key].append(name)

    for component, wireSets in ComponentMap.items():
        for wireSet in wireSets:
            addToMap(wireSet, component)
    
    return ConnectionMap

# Simulator
# _______________________________________________________________

# Utility function to get a socket from a socket name
def getSocket(socketName, socketMap):
    socket = next(
        (key for key in socketMap if key == socketName), 
        None)
    if socket is None:
        raise ValueError(f"Socket {socketName} not found in socket map")
    return socket


# Update is always flipping state of a source/emitter component/socket.
def UpdateCircuitSource(connectionMap, socketMap, source, depth):
    if (depth > 20):
        raise f"Maximum recursion depth of {depth} exceeded"
    
    # Update state of component
    source.state = not source.state

    for connection in socketMap[source]:
        socketsToUpdate = [socket for socket in connectionMap[connection] if not socket.isSource]

        if not source.state:
            # When source is turned off
            wireSources = [socket for socket in connectionMap[connection] if socket.isSource]

            if not any(source.state for source in wireSources):
                #If there is no source that is emitting a signal on the wireSet,
                # update connected components

                updateQueue = set() # Set of sources that are to be flipped.
                for socket in socketsToUpdate:
                    socket.state = False

                    prefix, comp_id, comp_type = socket.name

                    if comp_type:
                        emitterState = doesGateFire(
                            getSocket(prefix + comp_id + 'base', socketMap),
                            getSocket(prefix + comp_id + 'collector', socketMap),
                            prefix)

                        if emitterState != getSocket(prefix + comp_id + 'emitter', socketMap).state:
                            updateQueue.add(getSocket(prefix + comp_id + 'emitter', socketMap))

                for socket in updateQueue:
                    UpdateCircuitSource(connectionMap, socketMap, socket, depth + 1)
        else:
            # When source is turned on
            updateQueue = set() # Set of sources that are to be flipped.
            for socket in socketsToUpdate:
                socket.state = True

                prefix, comp_id, comp_type = socket.name

                if comp_type:
                    emitterState = doesGateFire(
                        getSocket(prefix + comp_id + 'base', socketMap),
                        getSocket(prefix + comp_id + 'collector', socketMap),
                        prefix)

                    if emitterState != getSocket(prefix + comp_id + 'emitter', socketMap).state:
                        updateQueue.add(getSocket(prefix + comp_id + 'emitter', socketMap))
            
            for socket in updateQueue:
                UpdateCircuitSource(connectionMap, socketMap, socket, depth + 1)


### ! WARNING ! ###
# Calling this function muliple times will change the result, because the sockets aren't copied
def Simulate(connectionMap, socketMap, operationOrder):

    # First, as input charges are defaulted to 0, activate NAND gates. Order could matter, but the Maps should be in the same order every time.
    for socket in socketMap:
        prefix, id, comp_type = socket.name

        if prefix == 'NAND' and not socket.state:
            UpdateCircuitSource(connectionMap, socketMap, socket, 0)

    for order in operationOrder:
        UpdateCircuitSource(connectionMap, socketMap, order, 0)

    #Return the final state of the circuit
    circuitState = []
    for socket in socketMap:
        prefix, id, comp_type = socket.name
        if not comp_type:
            circuitState.append((socket, socket.state))
    return circuitState