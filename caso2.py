import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from sim_visual import setup_axis, capture_frame, init_patches, save_frames
from sim_physics import resolve_collisions
import itertools


class Particula:

    def __init__(self,r,v,m,color,cont):
        # Instancias
        
        self.R = r
        self.V = v
        self.M = m
        self.Color = color
        self.Cont = cont

    def Pos_evol(self, t):
        # Evolucion temporal de la particula
        
        self.R = self.R+self.V*t

    def Colision(self, r2, v2):
        # Rebote en las colisiones
        
        r12 = self.R-r2
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

# Numero de particulas
n = 200
a = int(n/4)
b = n-a

# Posicion aleatoria
Xin1 = np.arange(-1+diametro,-0.5-3*radio,0.05)  
Xin2 = np.arange(-0.5+3*radio,1-diametro,0.05)
Yin = np.arange(-1+diametro,1-diametro,0.05)
puntos1 = np.array(list(itertools.product(Xin1,Yin)))
puntos2 = np.array(list(itertools.product(Xin2,Yin)))
particulas1 = puntos1[np.random.choice(len(puntos1), size=a, replace=False)]
particulas2 = puntos2[np.random.choice(len(puntos2), size=b, replace=False)]

# Velocidad aleatoria
vel = np.random.rand(n,2)*1-0.5

Par = []
Par.append(Particula(particulas1[0],vel[0],1,1,0))

for m in range(1,a):
    Par.append(Particula(particulas1[m],vel[m],1,0,0))

for p in range(a,n):
    Par.append(Particula(particulas2[p-a],vel[p],1,0,0))

colors = ["b","r","g"]      
images = []
target_shape_holder = {'shape': None}
tiempo = []
infectados = []
recuperados = []
t_total = 51
t_apertura = int(t_total/7)
delta_ap = 0.33/(int(t_total/2)-t_apertura)
apertura = delta_ap

for i in range(1,t_total):
    Par2 = Par
    Xs = []
    Ys = []
    Colores = []
    num_inf = 0
    num_rec = 200

    for j in range(n):
        Par2j = Par2[j]
        Parj = Par[j]
        Rj = Par2j.R
        Vj = Par2j.V

        # collisions will be resolved by central physics module
        pass

        # Condicion Paredes verticales    
        if (Rj[0] <= (-1+radio+delta_r)) or (Rj[0] >= (1-radio-delta_r)):
            Parj.ParedX()

        # Condicion Paredes horizontales
        if (Rj[1] <= (-1+radio+delta_r)) or (Rj[1] >= (1-radio-delta_r)):      
            Parj.ParedY()

        # 3 condiciones Pared aislante
        if (i < t_apertura and Rj[0] <= (-0.5+diametro)
                and Rj[0] >= (-0.5-diametro)):
            Parj.ParedX()

        if (i >= t_apertura and i <= int(t_total/2)
                and Rj[0] <= (-0.5+diametro) and Rj[0] >= (-0.5-diametro)
                and (Rj[1] >= (apertura-radio) or Rj[1] <= -(apertura-radio))):
            Parj.ParedX()

        if (i > int(t_total/2) and Rj[0] <= (-0.5+diametro)
                and Rj[0] >= (-0.5-diametro)
                and (Rj[1] >= (apertura-radio) or Rj[1] <= -(apertura-radio))):
            Parj.ParedX()

        if Parj.Color == 1:
            Parj.Cont += 1
            num_inf += 1

        # Tiempo de recuperacion    
        if Parj.Cont == 350:
            Parj.Color = 2

        if Parj.Color == 2:
            num_rec-=1
            
        Xs.append(Rj[0])
        Ys.append(Rj[1])
        Colores.append(Parj.Color)
        
        Parj.Pos_evol(0.02)  
        Par[j] = Parj

    # resolve all pairwise collisions for this step
    resolve_collisions(Par, diametro)

    if i >= t_apertura and i <= int(t_total/2):
        apertura+=delta_ap
        
    tiempo.append(i)
    infectados.append(num_inf)
    recuperados.append(num_rec)


# Codigo para crear y anhadir las graficas del gif

    fig, ax = setup_axis()
    patches = init_patches(ax, [type('P', (), {'R':[x,y], 'Color':c}) for x,y,c in zip(Xs,Ys,Colores)], radio, colors)
    # dibujar pared central según estado
    if i < t_apertura:
        ax.plot((-0.5,-0.5),(-1,1),'k')
    if i >= t_apertura and i <= int(t_total/2):
        ax.plot((-0.5,-0.5),(-1,-apertura),'k')
        ax.plot((-0.5,-0.5),(apertura,1),'k')
    if i > int(t_total/2):
        ax.plot((-0.5,-0.5),(apertura,1),'k')
        ax.plot((-0.5,-0.5),(-1,-apertura),'k')

    capture_frame(fig, images, target_shape_holder)
    plt.close()


# Graficacion de los resultados finales
# plt.figure()
# plt.fill_between(tiempo,0,infectados,facecolor='red',label='Contagiados')
# plt.fill_between(tiempo,recuperados,n,facecolor='green',label='Recuperados')
# plt.fill_between(tiempo,infectados,recuperados,facecolor='blue',label='Sanos')
# plt.ylim(0,n)
# plt.ylabel("Numero de personas")
# plt.xlabel("Tiempo")
# plt.title("Intento de cuarentena", loc='left')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig("Graficofinal2")
# plt.close()

# Generador del gif
imageio.mimsave('caso2.gif', images, 'GIF', duration=0.03)
