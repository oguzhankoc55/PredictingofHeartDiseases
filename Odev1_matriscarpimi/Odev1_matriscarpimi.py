def matriscarpimifonk(a,b,c,d):
    if (b != c):
        print("Matrislerin carpilmasi mumkun degil")
    else:
        
        X = [[0 for x in range(b)] for x in range(a)]
        Y = [[0 for x in range(d)] for x in range(c)]
        T = [[0 for x in range(d)] for x in range(a)]
        
        print("X matrisini giriniz:")
        for q in range(a):
            for w in range(b):
                print('X[{}][{}]'.format(q+1, w+1))
                X[q][w] = int(input())
        
        print("Y matrisini giriniz:")
        for q in range(c):
            for w in range(d):
                print('Y[{}][{}]'.format(q+1, w+1))
                Y[q][w] = int(input())
         
        for q in range(a):
            for w in range(d):
                for m in range(b):
                    T[q][w] += X[q][m] * Y[m][w]

        print(T)


print("Lütfen X(a,b) Boyutlarini girin:")
a= int(input())
b = int(input())
print("Lütfen Y(c,d) boyutlarini girin:")
c = int(input())
d = int(input())
matriscarpimifonk(a,b,c,d)
