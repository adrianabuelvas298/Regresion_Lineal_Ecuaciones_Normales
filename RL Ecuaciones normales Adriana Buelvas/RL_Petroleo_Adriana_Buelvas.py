import numpy as np
import numpy.linalg as linalg

y_entrenamiento = []    
y_pruebas = []
entradas_entrenamiento = []
datos_pruebas = [] 
h = []

Entradas_datos = []
def cargar_datos():
    with open('petrol_data.txt') as file:
        lineas = file.read().splitlines()
        for linea in lineas:
            matrix = []
            for data in linea.split(','):
                matrix.append(float(data))
            Entradas_datos.append(matrix)
    return Entradas_datos
valor=cargar_datos()
dataSet=np.array(valor)
x=dataSet[:,:4]
y=dataSet[:,4]

print("____________________________________________________________________")
print("|                   TALLER DE REGRESIÓN LINEAL E. NORMALES          |")
print("|___________________________________________________________________|\n")


n =  int((np.size(dataSet))/len(dataSet[0])) #Numero de filas

p_entrenamiento = round(n*0.60)
p_pruebas = n-p_entrenamiento

for i in range(0,p_entrenamiento):
    entradas_entrenamiento.append(x[i])
    y_entrenamiento.append(y[i])
entradas_entrenamiento = np.array(entradas_entrenamiento).reshape(np.shape(entradas_entrenamiento))
y_entrenamiento = np.array(y_entrenamiento).reshape(np.shape(y_entrenamiento))
entradas_entrenamiento = np.c_[np.ones(p_entrenamiento),entradas_entrenamiento]

#Aplicando funcón para calcular theta en ecuaciones normales

theta = (linalg.inv((entradas_entrenamiento.T)@entradas_entrenamiento)@(entradas_entrenamiento.T))@y_entrenamiento  


for i in range(p_entrenamiento,n):
    datos_pruebas.append(x[i])
    y_pruebas.append(y[i])
datos_pruebas = np.array(datos_pruebas).reshape(np.shape(datos_pruebas))
datos_pruebas = np.c_[np.ones(p_pruebas),datos_pruebas]
y_pruebas = np.array(y_pruebas).reshape(np.shape(y_pruebas))

#print("Valores el 40% para entrenamiento",datos_pruebas)
error = 0
for i in range(p_pruebas):
    model = theta[0]*datos_pruebas[i][0] + theta[1]*datos_pruebas[i][1] + theta[2]*datos_pruebas[i][2]+ theta[3]*datos_pruebas[i][3]+ theta[4]*datos_pruebas[i][4]
    h.append(model)
for i in range(p_pruebas):
    error = error + (abs((y_pruebas[i] - h[i]) / y_pruebas[i]))
error = (error/p_pruebas)*100


print("\n:::   GENERANDO  MODELO PARA EL  CONSUMO DE PETROLEO::\n")
print("El conjunto de los datos tiene:", len(Entradas_datos),"Elementos")
print("El número de elementos correspondientes al 60% de los datos es:" ,p_entrenamiento ,"Elementos")
print("El número de elementos correspondientes al 40% de los  datos es:",p_pruebas ,"Elementos")
print("\nLos valores de theta para la función del modelo h(x)  utilizando ecuaciones normales es::\n")
print("h(x):", theta)
print("El valor del MAPE para este modelo es:::"+str(error)+("%"))


    
