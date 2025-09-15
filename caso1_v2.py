import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
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

# Posicion aleatoria
Xin = np.arange(-1+2*radio+delta_r,1-2*radio-delta_r,0.05)
puntos = np.array(list(itertools.product(Xin,Xin)))
particulas = puntos[np.random.choice(len(puntos), size=n, replace=False)]

# Velocidad aleatoria
vel = np.random.rand(n,2)*1-0.5

Par = []

for m in range(n-1):
    Par.append(Particula(particulas[m],vel[m],1,0,0))

Par.append(Particula(particulas[n-1],vel[n-1],1,1,0))

colors = ["b","r","g"]
images = []
tiempo = []
infectados = []
recuperados = []

# Parámetros de animación y tiempo
dt = 0.02
num_steps = 80
frames = []
target_shape = None

# Preparar figura y parches
fig, ax = setup_axis()
patches = init_patches(ax, Par, radio, colors)

def update(frame_idx):
    global Par, target_shape
    # Resolver colisiones (módulo común)
    resolve_collisions(Par, diametro)

    # Rebotes en paredes, corrección posicional con paredes, actualización de estado y posiciones
    num_inf = 0
    num_rec = n
    for k in range(n):
        Rk = Par[k].R
        # paredes
        # comprobar e imponer corrección posicional si queda fuera de los límites
        if Rk[0] <= (-1 + radio):
            Par[k].R[0] = -1 + radio
            Par[k].V[0] = abs(Par[k].V[0])
        if Rk[0] >= (1 - radio):
            Par[k].R[0] = 1 - radio
            Par[k].V[0] = -abs(Par[k].V[0])
        if Rk[1] <= (-1 + radio):
            Par[k].R[1] = -1 + radio
            Par[k].V[1] = abs(Par[k].V[1])
        if Rk[1] >= (1 - radio):
            Par[k].R[1] = 1 - radio
            Par[k].V[1] = -abs(Par[k].V[1])
        # enfermedad
        if Par[k].Color == 1:
            Par[k].Cont += 1
            num_inf += 1
            if Par[k].Cont >= 550:
                Par[k].Color = 2
        if Par[k].Color == 2:
            num_rec -= 1
        # mover
        Par[k].Pos_evol(dt)
        # actualizar parche
        patches[k].center = (Par[k].R[0], Par[k].R[1])
        patches[k].set_color(colors[Par[k].Color])

    tiempo.append(frame_idx + 1)
    infectados.append(num_inf)
    recuperados.append(num_rec)


    # capturar frame en memoria usando util
    capture_frame(fig, frames, {'shape': None} if len(frames) == 0 else {'shape': target_shape})

    return patches

# Ejecutar animación en pantalla y guardar GIF al final
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=int(dt * 1000), 
                              blit=False, repeat=False)
plt.show()

# Guardar GIF con las frames capturadas
if frames:
    imageio.mimsave('caso1.gif', frames, duration=dt)
