from numpy import load

data = load('arithmetic_pkg/fidels_nQ=4_nL<=4.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])