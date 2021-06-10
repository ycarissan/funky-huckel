import numpy
import matplotlib.pyplot as plt

class Hamiltonien:
    def __init__(self):
        self.struct = []

    def addStructure(self, structure, num=1):
        for i in range(num):
            self.struct.append(structure)

    def populate(self):
        n=len(self.struct)
        self.data = numpy.zeros(shape=(n,n))
        for i in range(len(self.struct)):
            struct_i = self.struct[i]
            for j in range(len(self.struct)):
                struct_j = self.struct[j]
                if struct_i.nom in list(struct_j.couplage.keys()):
                    self.setData(i, j, struct_j.couplage[struct_i.nom])
        for i in range(len(self.struct)):
            struct = self.struct[i]
            self.setData(i,i,struct.energy)

    def solve(self):
        self.populate()
        return numpy.linalg.eig(self.data)

    def setData(self,i,j,k):
        self.data[i,j] = k
        self.data[j,i] = k

    def show(self):
        self.populate()
        print("Hamiltonien")
        print(self.data)

class Structure:
    def __init__(self, nom, energy, couplage):
        self.couplage={}
        self.nom = nom
        self.energy = energy
        self.couplage = couplage

def main():
    #fitted values on huckel:
    a=7.91141657
    b=2.80340421
    k=0.12705816
    #fitted values on DFT:
    a=-2.31788677e+02
    b=-7.66726086e+01
    k=-1.34646430e-02
    #a=0.8590
    #b=0.0744
    #k=0.3176
    
    #a=8
    #b=2
    #k=1
    
    k_I_II = k
    k_I_III = k
    k_II_III = k
    k_II_III_0=k_II_III
#    for a in numpy.arange(10,step=0.1):
    k_II_III=k_II_III_0
    structure_1 = Structure('I', 13*a+6*b, {'I':k, 'II':k, 'III':k})
    structure_2 = Structure('II', 10*a+11*b, {'II':k_II_III, 'III':k_II_III})
    structure_3 = Structure('III', 9*a+18*b, {'II':k_II_III, 'III':k_II_III})
    
    hamiltonien = Hamiltonien()
    hamiltonien.addStructure(structure_1)
    hamiltonien.addStructure(structure_2)
    hamiltonien.addStructure(structure_3,3)

    hamiltonien.show()
    eigvals, eigvects = hamiltonien.solve()

    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvects = eigvects[:,idx]
    print(a, eigvects[0])


    n=len(eigvals)

    fig, ax = plt.subplots(n,1)
    for i in range(n):
        ax[i].bar(range(n), [l*l for l in eigvects[i]])
        ax[i].set(title="{:.2f}".format(eigvals[i]))
        ax[i].grid()
    plt.show()



main()
