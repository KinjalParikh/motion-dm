import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import time
import colorsys

Spine = [0, 3, 6, 9, 12, 15]
Leg1 = [0, 1, 4, 7, 10]
Leg2 = [0, 2, 5, 8, 11]
Arm1 = [9, 13, 16, 18, 20, 22]
Arm2 = [9, 14, 17, 19, 21, 23]
body = [Spine, Leg2, Leg1, Arm2, Arm1]
edges = []

colours = np.zeros((23, 3))
# col = np.zeros((5, 3))
# col[1, :] = [1, 0, 0]
# col[2, :] = [0, 1, 0]
# col[3, :] = [0, 0, 1]
# col[4, :] = [0.5, 0.5, 0.25]

for l_ind in range(len(body)):
    col = colorsys.hsv_to_rgb(l_ind/5, 1, 0.75)
    for i in range(len(body[l_ind])-1):
        colours[len(edges), :] = col
        edges.append([body[l_ind][i], body[l_ind][i+1]])


batch_samples = np.load("../save/trial4/samples_trial4_000000933_seed42/results.npy")
denormalize = np.load("../dataset/meta.npy")


for i in range(3):
    mean = denormalize[i, 0]
    std = denormalize[i, 1]
    batch_samples[:, :, i, :] = -1.5 * ((batch_samples[:, :, i, :]*std)-mean)


s_ind = 8
sample_i = batch_samples[s_ind, :, :, :]
# sample_i = batch_samples

ps.init()
# nodes = sample_i[0, :, :]
nodes = sample_i[:, :, 0]
ps_net = ps.register_curve_network("human", nodes, np.array(edges))
ps_net.add_color_quantity("cols", colours, defined_on='edges', enabled=True)

i = 0


def callback():
    global i

    if i < 40:
        # new_pos = sample_i[i, :, :]
        new_pos = sample_i[:, :, i]
        ps_net.update_node_positions(new_pos)
        time.sleep(0.2)
        i += 1

    if psim.Button("Restart"):
        i = 0


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()