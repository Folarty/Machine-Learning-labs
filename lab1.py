
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
dt=np.dtype("f8,f8,f8,f8,U30")
data2=np.genfromtxt("iris.data",delimiter=",",dtype=dt)
# fonvwe u3 ondensreoc cmonbuo8
sepal_length = [] # Sepal Length
sepal_width = [] # sepal width
petal_length = [] # Petal Length
petal_width = [] # Petal width
# Bunonnnen 06x00 6ced Konnexyuu data2
for dot in data2:
    sepal_length.append(dot[0])
    sepal_width.append(dot[1])
    petal_length. append(dot[2])
    petal_width.append(dot[3])

# cmpoun epapuru no npoexuunm downer
# YoumsBoen, no Kaxdve 50 muno8 upuco8 udym nocnedobarensHo
plt.figure(1)

setosa, = plt.plot(sepal_length[:50],sepal_width[:50],'ro',label='Setosa')
versicolor, = plt.plot(sepal_length[50:100], sepal_width[50:100],'g^', label='Versicolor')
virginica, = plt.plot(sepal_length[100:150], sepal_width[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Sepal Length')

plt.ylabel("Sepal width")

plt.figure(2)
Setosa, = plt.plot(sepal_length[:50], petal_length[:50], 'ro', label='Setosa')

versicolor, = plt.plot(sepal_length[50:100], petal_length[50:100], 'g^', label='versicolor')
virginica, = plt.plot(sepal_length[100:150], petal_length[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel("Sepal Length")
plt.ylabel( 'Petal Length')

plt.figure(3)
setosa, = plt.plot(sepal_length[:50], petal_width[:50], 'ro', label='setosa')
versicolor, = plt.plot(sepal_length[50:100], petal_width[50:100], 'g^', label='Versicolor')
virginica, = plt.plot(sepal_length[100:150], petal_width[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Sepal Length")
plt.ylabel('Petal width')

plt.figure(4)
setosa, = plt.plot(petal_length[:50], petal_width[:50], 'ro', label='setosa')
versicolor, = plt.plot(petal_length[50:100], petal_width[50:100], 'g^', label='Versicolor')
virginica, = plt.plot(petal_length[100:150], petal_width[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Petal Length")
plt.ylabel('Petal width')
plt.show()