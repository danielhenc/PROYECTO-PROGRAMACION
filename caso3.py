import numpy as np
import matplotlib.pyplot as plt
import imageio
import itertools


class Particula:

    
    def __init__(self,r,v,m,color,cont,mov):
        # Instancias
        
        self.R = r
        self.V = v
        self.M = m
        self.Color = color
        self.Cont = cont 
        self.Mov = mov

    def Pos_evol(self, t):
        # Evolucion temporal de la particula
            
        self.R = self.R+self.V*t

    def Colision(self, r2, v2, mov2):

        r12 = self.R-r2

        # Rebote de particula en movimiento con particula quieta
        if mov2 == 0:
            self.V = self.V-2*np.dot(self.V,r12)*(r12)/((np.linalg.norm(r12))**2)

        # Rebote entre particulas en movimiento
        else:
            v12 = self.V-v2
            self.V = self.V-np.dot(v12,r12)*(r12)/((np.linalg.norm(r12))**2)
        
    def ParedX(self):
        # Rebote paredes verticales

        self.V[0] = -self.V[0]

    def ParedY(self):
        # Rebote paredes horizontales

        self.V[1] = -self.V[1]

# Tamanho de particulas
diametro = 0.03
radio = diametro/2.
delta_r = 0.004

#Numero de particulas
n = 200             
q = int(3*n/4)
s = int(n/4)

# Posicion aleatoria
Xin = np.arange(-1+2*radio+delta_r,1-2*radio-delta_r,0.05) 
puntos = np.array(list(itertools.product(Xin,Xin)))
particulas = puntos[np.random.choice(len(puntos), size=n, replace=False)]

# Velocidad aleatoria
vel = np.random.rand(s,2)*1-0.5

Par = []

for m in range(s-1):
    Par.append(Particula(particulas[m],vel[m],1,0,0,1))

Par.append(Particula(particulas[s-1],vel[s-1],1,1,0,1))
    
for p in range(s,n):
    Par.append(Particula(particulas[p],[0,0],1,0,0,0))
    

colors = ["b","r","g"] 
images = []
tiempo = []
infectados = []
recuperados = []

for i in range(1,1001):
    Par2 = Par
    Xs = []
    Ys = []
    Colores = []
    num_inf = 0
    num_rec = 200

    #Recorre las particulas en movimiento
    for j in range(s):
        Par2j = Par2[j]
        Parj = Par[j]
        
        Rj = Par2j.R
        Vj = Par2j.V
        
        for k in range(j+1,n):
            Par2k = Par2[k]
            Park = Par[k]

            #Condicion de colision
            if (np.linalg.norm(Par2j.R-Par2k.R) <= (diametro+delta_r)):
                
                if Parj.Color == 0 and Park.Color == 1:
                    Parj.Color = 1
                elif Parj.Color == 1 and Park.Color == 0:
                    Park.Color = 1

                #Revisando particulas en movimiento
                if Parj.Mov == 1:
                    Parj.Colision(Park.R,Park.V,Park.Mov)
                if Park.Mov == 1:
                    Park.Colision(Parj.R,Vj,Parj.Mov)

            Par[k] = Park

        # Condicion Paredes verticales 
        if (Rj[0] <= (-1+radio+delta_r)) or (Rj[0] >= (1-radio-delta_r)):
            Parj.ParedX()

        # Condicion Paredes horizontales
        if (Rj[1] <= (-1+radio+delta_r)) or (Rj[1] >= (1-radio-delta_r)): 
            Parj.ParedY()

        if Parj.Color == 1:
            Parj.Cont += 1
            num_inf += 1

        #Tiempo de recuperacion
        if Parj.Cont == 350:
            Parj.Color = 2

        if Parj.Color == 2:
            num_rec -= 1
            
        Xs.append(Rj[0])
        Ys.append(Rj[1])
        Colores.append(Parj.Color)

        Parj.Pos_evol(0.02)
        
        Par[j] = Parj

    #Recorre las particulas quietas
    for z in range(s,n):
        Parz = Par[z]
        Rz = Parz.R
        
        if Parz.Color == 1:
            Parz.Cont += 1
            num_inf += 1
            
        if Parz.Cont == 350:    
            Parz.Color = 2

        if Parz.Color == 2:
            num_rec -= 1
            
        Xs.append(Rz[0])
        Ys.append(Rz[1])
        Colores.append(Parz.Color)

        Par[z] = Parz

    tiempo.append(i)
    infectados.append(num_inf)
    recuperados.append(num_rec)

# Codigo para crear y anhadir las graficas del gif
'''    
    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.set(xlim=(-1,1),ylim=(-1,1))
    circles = [plt.Circle((xi,yi), radius=radio, color=colors[ci])
               for xi,yi,ci in zip(Xs,Ys,Colores)]
    
    for l in range(n):
        ax.add_artist(circles[l])
        
    plt.savefig("3Fig{0}".format(i))
    plt.close()

    images.append(imageio.imread("3Fig{0}.png".format(i)))
'''    

# Graficacion de los resultados finales
plt.figure()
plt.fill_between(tiempo,0,infectados,facecolor='red',label='Contagiados')
plt.fill_between(tiempo,recuperados,n,facecolor='green',label='Recuperados')
plt.fill_between(tiempo,infectados,recuperados,facecolor='blue',label='Sanos')
plt.ylim(0,n)
plt.ylabel("Numero de personas")
plt.xlabel("Tiempo")
plt.title("Distanciamiento moderado", loc='left')
plt.legend(loc=2)
plt.grid()
plt.savefig("Graficofinal3")
plt.close()

# Generador del gif
#imageio.mimsave('caso3.gif', images, 'GIF', duration=0.05)
