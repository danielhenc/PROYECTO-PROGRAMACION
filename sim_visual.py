import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def setup_axis(xlim=(-1,1), ylim=(-1,1), draw_walls=True):
    fig, ax = plt.subplots()
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    if draw_walls:
        wall = plt.Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0], fill=False, edgecolor='black', linewidth=1.5)
        ax.add_artist(wall)
    return fig, ax


def capture_frame(fig, frames, target_shape_holder):
    """Captura la figura en memoria y añade al array frames. target_shape_holder es un dict mutable para almacenar target_shape."""
    fig.canvas.draw()
    raw = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    pixels = raw.size // 4
    if pixels == w * h:
        buf = raw.reshape((h, w, 4))
    else:
        try:
            arr = np.asarray(fig.canvas.renderer.buffer_rgba())
            buf = arr
        except Exception:
            if pixels % w == 0:
                h_actual = pixels // w
                buf = raw.reshape((h_actual, w, 4))
            elif pixels % h == 0:
                w_actual = pixels // h
                buf = raw.reshape((h, w_actual, 4))
            else:
                raise RuntimeError(f"No se pudo determinar dimensiones del buffer: raw.size={raw.size}, get_width_height={(w,h)}")

    rgb = buf[:, :, 1:4]
    if target_shape_holder.get('shape') is None:
        target_shape_holder['shape'] = rgb.shape
        frames.append(rgb)
    else:
        target_shape = target_shape_holder['shape']
        if rgb.shape != target_shape:
            img = Image.fromarray(rgb)
            img = img.resize((target_shape[1], target_shape[0]), resample=Image.BILINEAR)
            frames.append(np.array(img))
        else:
            frames.append(rgb)


def init_patches(ax, particles, radio, colors):
    patches = [plt.Circle((p.R[0], p.R[1]), radius=radio, color=colors[p.Color]) for p in particles]
    for patch in patches:
        ax.add_artist(patch)
    return patches


def save_frames(frames, filename, duration=0.03):
    import imageio
    if frames:
        imageio.mimsave(filename, frames, duration=duration)
