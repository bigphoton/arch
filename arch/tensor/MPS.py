import numpy as np

"""Python file for definition of Matrix product state class"""

class MPS():
    """Class definition for Matrix product state"""


    def __init__(self,site_tensors):
        """Initialises an empty MPS defined by the number of site tensors"""

        """arrays that hold the bond and site tensors/connections"""
        self.site_tensors = []
        self.site_tensor_connections = []
        self.bond_tensors = []
        self.bond_tensor_connections = []

        """arrays for combined lists of site and bond tensors/connections"""
        self.tensors = []
        self.connections = []
        self.internal_link_indices = range(1,2*site_tensors+1)

        """make site tensors"""
        for i in range(1,site_tensors+1):
            self.site_tensors.append(np.zeros(shape=(1, 1, 1)))
            self.site_tensor_connections.append([-i, 2*i-1, 2*i])

        """make bond tensors"""
        for k in range(1,site_tensors+2):
            """edge cases"""
            if k == 1:
                self.bond_tensors.append(np.zeros(shape=1))
                self.bond_tensor_connections.append([1])
            elif k == site_tensors+1:
                self.bond_tensors.append(np.zeros(shape=1))
                self.bond_tensor_connections.append([2*site_tensors])
            else:
                """regular case"""
                self.bond_tensors.append(np.zeros(shape=(1,1)))
                self.bond_tensor_connections.append([2*(k-1), 2*(k-1)+1])

        """Combine into single tensor and connections list, potentially useful for future operations"""
        for t, site in self.site_tensors:

            self.tensors.append(self.bond_tensors[t])
            self.connections.append(self.bond_tensor_connections[t])

            self.tensors.append(self.site_tensors[t])
            self.connections.append(self.site_tensor_connections[t])

        """append last bond tensor"""
        self.tensors.append(self.bond_tensors[-1])
        self.connections.append(self.bond_tensor_connections[-1])


    def update_combined_tensors_connections(self):

        self.tensors = []
        self.connections = []

        """Combine into single tensor and connections list, potentially useful for future operations"""
        for t, site in self.site_tensors:
            self.tensors.append(self.bond_tensors[t])
            self.connections.append(self.bond_tensor_connections[t])

            self.tensors.append(self.site_tensors[t])
            self.connections.append(self.site_tensor_connections[t])

        """append last bond tensor"""
        self.tensors.append(self.bond_tensors[-1])
        self.connections.append(self.bond_tensor_connections[-1])

    def populate(self,site_fn,bond_fn):

        temp_site_tensors = []
        for i,site_tensor in enumerate(self.site_tensors):
            temp_site_tensors.append(site_fn(i+1))

        temp_bond_tensors = []
        for k, bond_tensor in enumerate(self.bond_tensors):
            temp_bond_tensors.append(bond_fn(k+1))

        self.site_tensors = temp_site_tensors
        self.bond_tensors = temp_bond_tensors

        self.update_combined_tensors_connections()

        return

    def branch_density_matrices(self,internal_link):

        """Returns the two branch density matrices after splitting at an internal link"""

        rho1_top_tensors = self.tensors[0:internal_link]
        rho1_bot_tensors = [np.conjugate(x) for x in rho1_top_tensors]
        rho1_top_connections = self.connections[0:internal_link]
        rho1_bot_connections = rho1_top_connections

        rho2_top_tensors = self.tensors[internal_link:]
        rho2_bot_tensors = [np.conjugate(x) for x in rho2_top_tensors]
        rho2_top_connections = self.connections[internal_link:]
        rho2_bot_connections = rho2_top_connections




        return

    def canonical_form(self):
        """Brings the MPS into canonical form where every link is a center of orthogonality"""
        return


