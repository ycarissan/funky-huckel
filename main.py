from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
from scipy import optimize
import matplotlib.pyplot as plt
import numpy
import tikzplotlib

def E_acene(a,b,k,x):
    return -(a + 2*b*(x-1) + 2*k*numpy.cos(numpy.pi/(x+1)) )

def func(p, x, y_to_fit):
    a=p[0]
    b=p[1]
    k=p[2]
    E = E_acene(a, b, k, x)
    delta = E-y_to_fit
    return delta

def getHuckelEnergy(ncycle):
    n=int(4*ncycle+2)
    data = numpy.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            elif numpy.abs(i-j)==1:
                data[i,j] = 1
            if i<n/2 and (i % 2) == 0 and i+j==n-1:
                data[i,j]=1
                data[j,i]=1
    eigv = numpy.array(numpy.linalg.eigvals(data))
    eigv = -numpy.sort(-eigv) # sort in reverse order
    etot = sum([2*eigv[i] for i in range(int(n/2))])
    return etot

def getDelocEnergy(ncycle):
    n=int(4*ncycle+2)
    data = numpy.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            elif numpy.abs(i-j)==1:
                data[i,j] = 1
    data[0,n-1]=1
    data[n-1,0]=1
    data[n//2, n//2-1]=1
    data[n//2-1, n//2]=1
    eigv = numpy.array(numpy.linalg.eigvals(data))
    eigv = -numpy.sort(-eigv) # sort in reverse order
    etot = sum([2*eigv[i] for i in range(int(n/2))])
    return etot

def getLocEnergy(ncycle):
    n=int(4*ncycle+2)
    etot = 8 + (n-6)/2
    return etot

def getHuckelEnergy_helix(ncycle):
    n=int(4*ncycle+2)
    data = numpy.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            elif numpy.abs(i-j)==1:
                data[i,j] = 1
    for i in range(2,n,3):
        j=int(((12*ncycle-1)-i)/3)
        data[i,j] = 1
        data[j,i] = 1
    eigv = numpy.array(numpy.linalg.eigvals(data))
    eigv = -numpy.sort(-eigv) # sort in reverse order
    etot = sum([2*eigv[i] for i in range(int(n/2))])
    return etot

def main():
    NMAX=20
    x=numpy.array(range(NMAX))
    huckel_e=[]
    deloc_e=[]
    loc_e=[]
    huckel_helix_e=[]
    for ncycle in range(NMAX):
        huckel_e.append(-getHuckelEnergy(ncycle))
        deloc_e .append(-getDelocEnergy(ncycle))
        loc_e .append(-getLocEnergy(ncycle))
        huckel_helix_e.append(-getHuckelEnergy_helix(ncycle))
        print("{} {} {} {} {}".format(ncycle,huckel_e[-1], deloc_e[-1], loc_e[-1], huckel_helix_e[-1]))

    loc_huckel_e = numpy.array(loc_e)-numpy.array(huckel_e)

    c = polyfit(x, huckel_e, 1)
    x_ = x
    y_ = polyval(x_, c)

    b = -c[1]/2
    a = c[0]+c[1]
    print("a= {}".format(a))
    print("b= {}".format(b))

    p0 = [a, b, 0.01]
    p1, success = optimize.leastsq(func, p0, args=(x,huckel_e))
    a=p1[0]
    b=p1[1]
    k=p1[2]
    print(p1, success)
    E_acene_vals = [E_acene(a, b, k, x[i]) for i in range(len(x))]

    # Data for plotting
    fig, ax = plt.subplots(3,1)
    ax[0].scatter(x, huckel_e, label='Huckel')
    ax[0].scatter(x, deloc_e, label='Deloc')
    ax[0].scatter(x, loc_e, label='Loc')
    ax[0].plot(x, huckel_e)
    ax[0].plot(x, deloc_e)
    ax[0].plot(x, loc_e)
    ax[0].plot(x_, y_, label='fit')
    ax[0].set(xlabel='Number of cycle', ylabel='Energy(beta)',
           title='Energies in terms of the number of cycle')
    ax[0].grid()
    ax[0].legend()
    
    ax[1].scatter(x, [huckel_e[i]-E_acene_vals[i] for i in range(len(huckel_e))], label='diff')
    ax[1].set(xlabel='Number of cycle', ylabel='diff(beta)',
           title='Energy difference in terms of the number of cycle')
    ax[1].grid()
    ax[1].legend()

    ax[2].scatter(x, huckel_helix_e, label='Helice')
    ax[2].scatter(x, huckel_e, label='Acene')
    ax[2].plot(x, huckel_helix_e)
    ax[2].plot(x, huckel_e)
    ax[2].set(xlabel='Number of cycle', ylabel='Energy(beta)',
           title='Energies of helicene in terms of the number of cycle')
    ax[2].grid()
    ax[2].legend()
    
    fig.savefig("output.png")
    plt.show()
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(x, huckel_e, label='Huckel')
    ax.plot(x, huckel_e)
    ax.plot(x_, y_, label='fit')
    ax.set(xlabel='Number of cycle', ylabel='Energy(beta)',
        title='Energies in terms of the number of cycle')
    ax.grid()
    ax.legend()
    ax.set_xticks([0, 4, 9, 14, 19])
    ax.set_xticklabels(["1", "5", "10", "15", "20"])
    fig.savefig("huckel.png", format='png')

    tikzplotlib.save("huckel.tex")

if __name__=="__main__":
    main()

