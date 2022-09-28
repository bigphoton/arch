from toolbox import *

if __name__ == '__main__':
    Length = 10.
    space_steps = 20
    dz = Length/space_steps
    print(dz)
    photon_cutoff = 4
    alpha = 1.
    top = coherent_guassian_MPS(alpha=alpha, dz=dz,photon_cutoff=photon_cutoff)
    D = dispersion_operator(dt=0.1,dz=dz,photon_cutoff=photon_cutoff)

    # print(len(top[0]))
    #
    # sBA_left = top[0][50]
    # A = top[0][51]
    # sAB = top[0][52]
    # B = top[0][53]
    # sBA_right = top[0][54]
    #
    #
    # result = apply_gate_MPS(D,A, sAB, B, sBA_left, sBA_right)
    # print(result[0].shape)


    time_steps = 2

    final_state = dispersion_evolve(top,D,time_steps=time_steps)




    for t in range(time_steps):

        data = []
        for i in range(1,space_steps):
            start_index = 3*space_steps +1
            bot_temp = coherent_guassian_MPS(alpha=alpha, dz=dz, starting_index=start_index,photon_cutoff=photon_cutoff)
            top = [final_state[1][t][0],final_state[1][t][1]]


            conjugates = []
            for matrix in top[0]:
                conjugates.append(np.conjugate(matrix))

            bot = [conjugates,bot_temp[1]]



            tensors = []
            connections = []
            intensity = intensity_operator(top_index=3*(i+1)-1, bottom_index=3*(i+1)-1 + start_index,photon_cutoff=photon_cutoff)
            Identity = [np.ones(shape=(photon_cutoff, photon_cutoff)), [3*(i+1)-1 + start_index, 3*(i+1)-1]]
            #intensity = Identity

            tensors = top[0] + bot[0] + [intensity[0]]
            connections = top[1] + connection_mapping(bot[1], reindex=start_index, contraction_index=3*(i+1)-1) + [intensity[1]]

            contraction_order = []
            for i,edges in enumerate(top[1]):
                for e in edges:
                    if e not in contraction_order:
                        contraction_order.append(e)
                for x in bot[1][i]:
                    if x not in contraction_order:
                        contraction_order.append(x)




            ans = nc.ncon(tensors, connections,contraction_order)
            data.append(abs(ans))


        plt.plot(range(1,space_steps),data)

    plt.show()
