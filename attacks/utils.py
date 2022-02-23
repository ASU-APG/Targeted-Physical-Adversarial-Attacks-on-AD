# size of perturbation
PERTURBATION_SIZE = (25, 30)  # (x, y)


def get_target_state(directory, scenario):
    target_state_file = f'{directory}/scenario_{scenario}.npy'
    import numpy as np
    return np.load(target_state_file)


def get_perturbation_file_path(directory, scenario, T, eps):
    perturbation_dir = f'{directory}/scenario_{scenario}'
    from os import mkdir
    from os.path import exists
    if not exists(perturbation_dir):
        mkdir(perturbation_dir)
    return f'{perturbation_dir}/perturb_T_{T}_eps_{eps}.npz'


def save_loss_to_file(loss, directory, scenario, T, eps):
    perturbation_dir = f'{directory}/scenario_{scenario}'
    loss_file = f'{perturbation_dir}/loss_T_{T}_eps_{eps}.txt'
    f = open(loss_file, 'w')
    f.write(f'{round(loss, 3)}')
    f.close()


def read_loss_from_file(directory, scenario, T, eps):
    perturbation_dir = f'{directory}/scenario_{scenario}'
    loss_file = f'{perturbation_dir}/loss_T_{T}_eps_{eps}.txt'
    from os.path import exists
    if not exists(loss_file):
        return
    loss = None
    with open(loss_file) as f:
        for line in f.readlines():
            loss = float(line)
    return loss


def create_animation_video(directory, states_clean, states_adv, T, eps):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

    t = states_adv
    c = states_clean

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title("Clean Behavior")
    ax2.set_title("Attack Behavior")
    ax1.set_axis_off()
    ax2.set_axis_off()
    im1 = ax1.imshow(c[0].squeeze()[3], cmap='gray', animated=True)
    im2 = ax2.imshow(t[0].squeeze()[3], cmap='gray', animated=True)

    def data():
        i = 0
        while i < T - 1:
            i += 1
            yield i

    def update(data):
        i = data
        # ax1.set_title(f"Clean State at t = {i}")
        # ax2.set_title(f"Adv State at t = {i}")
        im1.set_array(c[i].squeeze()[3])
        im2.set_array(t[i].squeeze()[3])
        i += 1
        return im1, im2

    ani = animation.FuncAnimation(fig, update, np.arange(0, T), interval=100, blit=False, repeat=False)
    ani.save(f'{directory}/video_T_{T}_eps_{eps}.mp4')
    plt.close()


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    import math
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
