import numpy

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

def main():
    for ncycle in range(10):
        print("{} {}".format(ncycle,getHuckelEnergy(ncycle)))

if __name__=="__main__":
    main()
