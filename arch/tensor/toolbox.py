import numpy as np
import math
import ncon as nc
import matplotlib.pyplot as plt
import scipy.linalg as sp
from scipy.special import factorial

#Attempting to re create the nonlinear quantum optics paper

#first step define the MPS for an inital coherent state with some envelope

#define creation and anihilation operators
def creation_op(photon_cutoff=10):
    a = np.zeros((photon_cutoff,photon_cutoff))
    for i in range(photon_cutoff-1):
        a[i,i+1] = np.sqrt(i+1)
    return a

#define creation and anihilation operators
def anihilation_op(photon_cutoff=10):
    a = np.zeros((photon_cutoff,photon_cutoff))
    for i in range(photon_cutoff-1):
        a[i+1,i] = np.sqrt(i+1)
    return a

def guassian_amplitude(index,photons,alpha,Length=10,dz=0.1,sigma=1):
    x = index*dz - Length/2.
    f_m = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2. * (x**2)/(sigma**2))
    amp = np.exp(-1/2. * (abs(f_m*alpha))**2) * (np.sqrt((f_m*alpha)**(2*photons)/factorial(photons,exact=True)))
    return amp


def coherent_guassian_MPS(alpha,Length=10,dz=0.1,photon_cutoff=10,sigma=1, starting_index=0):
    """Retunns the tensors and connections describing a coherent pulse with Guassian envolope"""

    num_tensors = int(np.floor(Length / dz))

    #define the matrix and connections of each tensor
    tensors = []
    connections = []

    start = np.ones(shape=1)
    start_connection = [2+starting_index]
    tensors.append(start)
    connections.append(start_connection)

    for i in range(num_tensors):
        T = np.ndarray(shape=(photon_cutoff,1,1))
        for k in range(photon_cutoff):
            T[k,0,0] = guassian_amplitude(i,photons=k,alpha=alpha,Length=Length,dz=dz,sigma=sigma)

        if i == 0:
            connection = [1 + starting_index,2 + starting_index ,3 + starting_index ]
        else:
            connection = [3*(i+1) - 1 + starting_index, 3*(i+1) - 2 + starting_index , 3*(i+1) + starting_index]


        tensors.append(T)
        connections.append(connection)

        if i != num_tensors-1:
            lamda = np.ones(shape=(1,1))
            tensors.append(lamda)
            lamda_connection = [connection[2],connection[2]+1]
            connections.append(lamda_connection)

    end = np.ones(shape=1)
    end_connection = [3*num_tensors + starting_index]
    tensors.append(end)
    connections.append(end_connection)


    return tensors, connections

def intensity_operator(top_index,bottom_index,photon_cutoff=10):

    T =  anihilation_op(photon_cutoff=photon_cutoff) @  creation_op(photon_cutoff=photon_cutoff)
    connection = [bottom_index,top_index]

    return T, connection

def phase_operator(dt,photon_cutoff=10,dz=0.1):
    return sp.expm((1j*dt)/(2*dz) * creation_op(photon_cutoff) @ creation_op(photon_cutoff) @ anihilation_op(photon_cutoff) @ anihilation_op(photon_cutoff))

def dispersion_operator(dt,photon_cutoff=10,dz=0.1):
    A = np.tensordot(creation_op(photon_cutoff),anihilation_op(photon_cutoff),axes=0)
    B = np.tensordot(anihilation_op(photon_cutoff),creation_op(photon_cutoff),axes=0)
    C = np.tensordot(creation_op(photon_cutoff) @ anihilation_op(photon_cutoff), np.identity(photon_cutoff),axes=0)
    D = np.tensordot(np.identity(photon_cutoff), creation_op(photon_cutoff) @ anihilation_op(photon_cutoff),axes=0)

    sum = (A + B - C - D)
    sum = np.reshape(sum,(photon_cutoff**2,photon_cutoff**2))
    D = sp.expm((1j*dt/(2*(dz**2))) * sum)
    D = np.reshape(D,newshape=(photon_cutoff,photon_cutoff,photon_cutoff,photon_cutoff))
    return D

def intensity_MPO(Length=10,dz=0.1,photon_cuttoff=10):
    """Returns the MPO respresenting a measuremnt of the intenisty
    connections made to interface with coherent spate mps, will make general later need index mapping funtion"""
    num_tensors = int(np.floor(Length/dz))


    return

def connection_mapping(input_connections,reindex,contraction_index):
    """fixed for now will make general"""
    for i, connection in enumerate(input_connections):
        if len(connection) == 3 and connection[0] != contraction_index + reindex:
            input_connections[i] = [connection[0] - reindex, connection[1], connection[2]]

    return input_connections

def apply_gate_MPS(gateAB, A, sAB, B, sBA_left, sBA_right, chi=70, stol=1e-10):
  """ apply a gate to an MPS across and a A-B link. Truncate the MPS back to
  some desired dimension chi"""

  # ensure singular values are above tolerance threshold
  sBA_trim_left = sBA_left * (sBA_left > stol) + stol * (sBA_left < stol)
  sBA_trim_right = sBA_right * (sBA_right > stol) + stol * (sBA_right < stol)
  # contract gate into the MPS, then deompose composite tensor with SVD
  d = A.shape[0]
  chiBA_left = sBA_trim_left.shape[0]
  chiBA_right = sBA_trim_right.shape[0]


  C = np.diag(sBA_trim_left)
  C_x = np.zeros(shape=(C.shape[0],C.shape[0]))
  np.fill_diagonal(C_x,C)

  D = np.diag(sAB)
  D_x = np.zeros(shape=(D.shape[0], D.shape[0]))
  np.fill_diagonal(D_x, D)


  #print(sBA_trim_right)
  #print(sBA_trim_right.shape)
  E = np.diag(sBA_trim_right)
  E_x = np.zeros(shape=(E.shape[0], E.shape[0]))
  np.fill_diagonal(E_x, E)


  tensors = [C_x, A, D_x, B, E_x, gateAB]

  connects = [[-1, 2], [1, 2, 3], [3, 4], [5, 4, 6], [6,-2], [-3, -4, 1, 5]]

  nshape = [d * chiBA_left, d * chiBA_right]
  utemp, stemp, vhtemp = np.linalg.svd(nc.ncon(tensors, connects).reshape(nshape),
                                full_matrices=False)

  # truncate to reduced dimension
  chitemp = min(chi, len(stemp))
  utemp = utemp[:, range(chitemp)].reshape(sBA_trim_left.shape[0], d * chitemp)
  vhtemp = vhtemp[range(chitemp), :].reshape(chitemp * d, sBA_trim_right.shape[0])


  #temp diaged right/left BA matrices

  temp_one_over_left = np.diag(1 / sBA_trim_left)
  one_over_diag_sBA_trim_left = np.zeros(shape=(temp_one_over_left.shape[0],temp_one_over_left.shape[0]))
  np.fill_diagonal(one_over_diag_sBA_trim_left,temp_one_over_left)

  temp_one_over_right = np.diag(1 / sBA_trim_right)
  one_over_diag_sBA_trim_right = np.zeros(shape=(temp_one_over_right.shape[0], temp_one_over_right.shape[0]))
  np.fill_diagonal(one_over_diag_sBA_trim_right, temp_one_over_right)


  # remove environment weights to form new MPS tensors A and B
  A = (one_over_diag_sBA_trim_left @ utemp).reshape(d,sBA_trim_left.shape[0], chitemp)
  B = (vhtemp @ one_over_diag_sBA_trim_right).reshape(d, chitemp, sBA_trim_right.shape[0])


  # new weights
  sAB = stemp[range(chitemp)] / np.linalg.norm(stemp[range(chitemp)])
  dim = sAB.shape[0]
  sAB_temp = np.zeros((dim,dim))
  np.fill_diagonal(sAB_temp,sAB)


  return A, sAB_temp, B


def QP_applygate(A, B, D, gate,error_tol=1E-2,bond_dimension=70):

    """My version of applying a two site gate to tensors A,B and D"""

    tensors = [A,B,D,gate]

    connects = [[1,-1,3],[2,4,-2],[3,4],[1,2,-3,-4]]

    contracted_shape = [A.shape[1]*gate.shape[2],B.shape[2]*gate.shape[3]]

    contracted_tensor = nc.ncon(tensors,connects).reshape(contracted_shape)

    U, S, V = sp.svd(contracted_tensor)

    print("Shapes of SVD tensors U S V:",U.shape,S.shape,V.shape)

    # Can now truncate the decomposition by throwing away singular values below the tolerance threashold
    # we sum the squares in the reverse order deleting rowa/columns for U and V as we go until the threashold is reached

    e_2 = error_tol**2
    squares_sum = 0.
    rc_to_del = 0

    for i in reversed(range(len(S))):
        squares_sum += S[i]**2

        if squares_sum < e_2:
            # Now delete rows/cols of U/V respectivly
            rc_to_del += 1

    rows_to_keep = len(S) - rc_to_del

    U_prime = U[:,:rows_to_keep]#np.delete(U,rc_to_del,1)
    V_prime = V[:rows_to_keep,:]#np.delete(V,rc_to_del,0)
    S_prime = S[0:len(S)-rc_to_del]
    temp = np.zeros((len(S_prime), len(S_prime)))
    np.fill_diagonal(temp, S_prime)

    #Now just reshape the new A, B and D tensors

    A_prime = U_prime.reshape((gate.shape[2],A.shape[1],len(S_prime)))
    D_prime = temp
    B_prime = V_prime.reshape((gate.shape[3],len(S_prime),B.shape[2]))

    #TODO thinking may need to normalise A and B trying that now for testing

    # A_norm = np.sqrt(nc.ncon([A_prime,np.conjugate(A_prime)],[[1,2,3],[1,2,3]]))
    # B_norm = np.sqrt(nc.ncon([B_prime,np.conjugate(B_prime)],[[1,2,3],[1,2,3]]))
    #
    # A_prime = A_prime / A_norm
    # B_prime = B_prime / B_norm

    return A_prime, D_prime, B_prime


def apply_dispersion(MPS,D,kind='even'):
    tensors = MPS[0]

    #print("total number of tensors is {}".format(len(tensors)))

    connections = MPS[1]
    num_D = int(len(tensors)/4)

    output_tensors = []

    if kind == 'even':
        for i in range(num_D):
            sBA_left = tensors[4 * i]
            A = tensors[4 * i + 1]
            sAB = tensors[4 * i + 2]
            B = tensors[4 * i + 3]
            sBA_right = tensors[4 * i + 4]

            # TESTING: trying my application of gate
            #result = apply_gate_MPS(gateAB=D,A=A,sAB=sAB,B=B,sBA_left=sBA_left,sBA_right=sBA_right)

            print("shapes of tensors A, B, D, gate in apply gate:",A.shape,B.shape,sAB.shape,D.shape)

            result = QP_applygate(A=A,B=B,D=sAB,gate=D)

            output_tensors.append(sBA_left)
            output_tensors.append(result[0])
            output_tensors.append(result[1])
            output_tensors.append(result[2])
            if i == num_D-1:
                output_tensors.append(sBA_right)

    elif kind == 'odd':
        # add in edge tensors not acted on by odd time step

        output_tensors.append(tensors[0])
        output_tensors.append(tensors[1])
        for i in range(num_D-1):
            sBA_left = tensors[4 * i +2]
            A = tensors[4 * i + 1 +2]
            sAB = tensors[4 * i + 2 +2]
            B = tensors[4 * i + 3 +2]
            sBA_right = tensors[4 * i + 4 +2]

            #print("shape of sbaleft is{} A is {} sab is {} b is {} sbaright is {}".format(sBA_left.shape,A.shape,sAB.shape,B.shape,sBA_right.shape))


            # TESTING: trying my application of gate
            # result = apply_gate_MPS(gateAB=D, A=A, sAB=sAB, B=B, sBA_left=sBA_left, sBA_right=sBA_right)
            result = QP_applygate(A=A, B=B, D=sAB, gate=D)

            output_tensors.append(sBA_left)
            output_tensors.append(result[0])
            output_tensors.append(result[1])
            output_tensors.append(result[2])
            if i == num_D - 2:
                output_tensors.append(sBA_right)

        output_tensors.append(tensors[-2])
        output_tensors.append(tensors[-1])

    return output_tensors, connections


def dispersion_evolve(initial_MPS,Dispersion,time_steps=10):

    state = initial_MPS
    state_time_steps = [initial_MPS]

    for i in range(time_steps):
        print("#######################################################################")
        # if i % 2 ==0:
        #     state = apply_dispersion(state,Dispersion,'even')
        # else:
        #     state = apply_dispersion(state,Dispersion,'odd')

        state = apply_dispersion(state, Dispersion, 'even')
        state = apply_dispersion(state, Dispersion, 'odd')

        # #TODO trying renormalisation of the state
        # tensors = state[0] + [np.conjugate(x) for x in state[0]]
        # state2_con = connection_mapping(state2[1], reindex=150, contraction_index=0)
        # connections = state1[1] + state2_con


        state_time_steps.append(state)

    return state, state_time_steps

