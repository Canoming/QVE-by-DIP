# coding: utf-8

import numpy as np
import random
from qutip import *

# Constants
r2 = np.sqrt(.5);
Gate = {
    'I' :qeye(2),
    'X': snot(),
    'Y': Qobj([[r2,r2*1.j],[r2*1.j,r2]]),
    'Z': qeye(2)
}


def H2RandBit(Hami, Probability):
    # Convert Hamiltonina to (Binary String, Pauli String)
    
    h = np.random.choice(Hami,p = Probability); # Choose one Pauli string
    # according to the probability.

    post = []; # indexes of non-identity position
    
    Str = h['String']; # The chosen Pauli string
    rbit = []; # Rand bits
    
    # Record non-identity position
    for i in range(len(Str)):
        if Str[i] != 'I':
            post.append(i);
        
    # Generate random numbers
    for i in range(len(Str)):
        rbit.append(random.getrandbits(1));
        
    sgn = bool(h['Sign'])^bool(1); # sgn = 1 if negative

    if len(post):
        if len(post)>1:
            for k in post:
                sgn = bool(sgn)^bool(rbit[k]);
            rbit[post[0]] = int(bool(rbit[post[0]])^bool(sgn));
        else:
            rbit[post[0]] = int(sgn);
    
    return [rbit,Str];


def VCirc(Parameters, Dimension):
    # Generate the variational circuit
    qc = QubitCircuit(Dimension)
    
    for i in range(Dimension):
        qc.add_gate("RZ", i, None, Parameters[0])
        qc.add_gate("RX", i, None, Parameters[1])
        qc.add_gate("RY", i, None, Parameters[2])
    return qc

def InvVCirc(Parameters, Dimension):
    # Generate the inverse of variational circuit
    qc = QubitCircuit(Dimension)
    
    for i in range(Dimension):
        qc.add_gate("RY", i, None, -Parameters[2])
        qc.add_gate("RX", i, None, -Parameters[1])
        qc.add_gate("RZ", i, None, -Parameters[0])
    return qc


def DIPCirc(Parameters, Dimension, vcirc=VCirc):
    # The wrap U(theta) with DIP test circuit
    base_circ =  QubitCircuit(2*Dimension);
    base_circ.add_circuit(vcirc(Parameters,Dimension),0);
    base_circ.add_circuit(vcirc(Parameters,Dimension),Dimension);

    for i in range(Dimension):
        base_circ.add_gate("CNOT", i, i+Dimension, None)
    
    # Output matrix instead of circuit
    U = gate_sequence_product(base_circ.propagators())
    return U


def StringPre(String,N):
    # Convert binary string to state
    Hstring = fock(np.power(2,N),int("".join(str(x) for x in String), 2))
    Hstring.dims = [[2]*N,[1]*N]
    return Hstring


def PauliPre(String,N,gate):
    # Convert Pauli String to Gates
    temp = Qobj(1);
    for i in range(N):
        temp = tensor(temp,gate[String[i]])
        temp.dims=[[2]*N,[2]*N]
    return temp;


def DIP(Para, N, Hamiltonian, Probability, SampleSize):
    
    # Generate Random Sample
    Inputs = []
    for i in range(SampleSize):
        Inputs.append([H2RandBit(Hamiltonian,Probability),H2RandBit(Hamiltonian,Probability)]);
    
    Upara = DIPCirc(Para,N);
    Output = [];
    
    
    for PauStr in Inputs:
        Str0 = StringPre(PauStr[0][0],N);
        PreOp0 = PauliPre(PauStr[0][1],N,Gate);
        Input0 = PreOp0*Str0;
        
        Str1 = StringPre(PauStr[1][0],N);
        PreOp1 = PauliPre(PauStr[1][1],N,Gate);
        Input1 = PreOp1*Str1;
        
        Input = tensor(Input0,Input1);
        
        Final_State = Upara*Input;
        Result = (Final_State*Final_State.dag()).ptrace([1]);
        
        Output.append(Result.data[(0,0)]);
#     vari = np.var(Output);
    return -np.abs(np.mean(Output))

