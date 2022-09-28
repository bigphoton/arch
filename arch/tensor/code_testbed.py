import numpy as np
import ncon as nc
import scipy.linalg as sp
from scipy.special import factorial
from tensor_tests import dispersion_operator
from toolbox import *
from visual import *


def guassian_amplitude(index,photons,alpha,Length=10,dz=0.1,sigma=1):
    x = index*dz - Length/2.
    f_m = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2. * (x**2)/(sigma**2))

    amp = np.exp(-1/2. * (abs(f_m*alpha))**2) * (np.sqrt((f_m*alpha)**(2*photons)/factorial(photons,exact=True)))
    return amp

def coherent_guassian_tensor_pair(index,alpha=1000.,Length=10,dz=0.01,photon_cutoff=3,sigma=1):
    """Retunns the tensors and connections describing a coherent pulse with Guassian envolope"""



    A = np.ndarray(shape=(photon_cutoff, 1, 1))
    for k in range(photon_cutoff):
        A[k, 0, 0] = guassian_amplitude(index, photons=k, alpha=alpha, Length=Length, dz=dz, sigma=sigma)


    lamda = np.ones(shape=(1, 1))

    B = np.ndarray(shape=(photon_cutoff, 1, 1))
    for k in range(photon_cutoff):
        B[k, 0, 0] = guassian_amplitude(index+1, photons=k, alpha=alpha, Length=Length, dz=dz, sigma=sigma)


    return A, lamda, B




def QP_apply2_singlesite(A, B, D, gate1,gate2,error_tol=1E-5,bond_dimension=70):

    """My version of applying a two site gate to tensors A,B and D"""

    tensors = [A,B,D,gate1,gate2]

    connects = [[1,-1,3],[2,4,-2],[3,4],[1,-3],[2,-4]]

    contracted_shape = [A.shape[1]*gate1.shape[1],B.shape[2]*gate2.shape[1]]

    contracted_tensor = nc.ncon(tensors,connects).reshape(contracted_shape)

    print(contracted_tensor)
    print("contracted tensor eigs:",sp.eigvals(contracted_tensor))

    U, S, V = sp.svd(contracted_tensor)

    return U,S,V


def QP_applygate(A, B, D, gate,error_tol=1E-5,bond_dimension=70):

    """My version of applying a two site gate to tensors A,B and D"""

    tensors = [A,B,D,gate]

    connects = [[1,-1,3],[2,4,-2],[3,4],[1,2,-3,-4]]

    contracted_shape = [A.shape[1]*gate.shape[2],B.shape[2]*gate.shape[3]]

    contracted_tensor = nc.ncon(tensors,connects).reshape(contracted_shape)

    U, S, V = sp.svd(contracted_tensor)

    # Can now truncate the decomposition by throwing away singular values below the tolerance threashold
    # we sum the squares in the reverse order deleting rowa/columns for U and V as we go until the threashold is reached

    e_2 = error_tol**2
    squares_sum = 0.
    rc_to_del = []

    for i in reversed(range(len(S))):
        squares_sum += S[i]**2
        print("squred_sum:",squares_sum)

        if squares_sum < e_2:
            # Now delete rows/cols of U/V respectivly
            rc_to_del.append(i)
        else:
            break

    print("del list:", rc_to_del)

    print("U:",U)
    print("##################")
    print("S:",S)
    print("##################")
    print("V:",V)

    U_prime = np.delete(U,rc_to_del,1)
    V_prime = np.delete(V,rc_to_del,0)
    S_prime = S[0:len(S)-len(rc_to_del)]
    temp = np.zeros((len(S_prime), len(S_prime)))
    np.fill_diagonal(temp, S_prime)

    #Now just reshape the new A, B and D tensors

    A_prime = U_prime.reshape((gate.shape[2],A.shape[1],len(S_prime)))
    D_prime = temp
    B_prime = V_prime.reshape((gate.shape[3],len(S_prime),B.shape[2]))

    return A_prime, D_prime, B_prime


def overlap(MPS,connections):
    """Calculates teh overlap of an MPS with itself"""
    tensors = MPS[0] + [np.conjugate(x) for x in MPS[0]]

    return





state1 = coherent_guassian_MPS(alpha=1.,dz=.2,photon_cutoff=3)


draw_graph(state1)

# state2 = coherent_guassian_MPS(alpha=1.,dz=.2,starting_index=150)
#
#
# tensors = fstate[0] + [np.conjugate(x) for x in fstate[0]]
# state2_con = connection_mapping(state2[1],reindex=150,contraction_index=0)
# connections = state1[1] + state2_con
#
#
# contraction_order = []
# for i,edges in enumerate(state1[1]):
#     for e in edges:
#         if e not in contraction_order:
#             contraction_order.append(e)
#     for x in state2_con[i]:
#         if x not in contraction_order:
#             contraction_order.append(x)
#
#
# ans = np.sqrt(nc.ncon(tensors, connections,contraction_order))
#
# print(ans)


