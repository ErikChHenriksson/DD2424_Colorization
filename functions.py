import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from scipy import spatial


def getQ(pixels):
    colors = np.zeros((22, 22))

    for p in pixels:
        a, b = p
        colors[get_index(a), get_index(b)] = 1

    return np.count_nonzero(colors)


def get_index(num):
    return (num + 110) / 10


def ab_to_quantization(ab):
    a, b = ab
    return math.floor(get_index(a)), math.floor(get_index(b))


def quantization_to_ab(abq):
    aq, bq = abq
    a = math.floor((aq * 10)-110)
    b = math.floor((bq * 10)-110)
    return a, b


# def one_hot_quantization(ab):
#     h, w, _ = ab.shape
#     ab_one_hot = np.zeros((h, w, 22, 22))
#     for a in range(h):
#         for b in range(w):
#             one_hot = np.zeros((22, 22))
#             aq, bq = ab_to_quantization((a, b))
#             one_hot[aq, bq] = 1
#             ab_one_hot[a, b, :, :] = one_hot

#     return ab_one_hot


def one_hot_quantization(ab, q_list):
    h, w, _ = ab.shape
    q = q_list.shape[0]
    ab_one_hot = np.zeros((h, w, q))
    ab = np.where(q_list == ab)
    ab_one_hot[ab[0], ab[1]] = 1

    return ab_one_hot


def space_to_points():
    space = np.load('./quantized_space/q_space.npy')
    points = np.zeros((space.shape[0], space.shape[1], 2))
    for i in range(space.shape[0]):
        for j in range(space.shape[1]):
            points[i, j] = [space[i, j]*i*10, space[i, j]*j*10]
    return points.flatten()


def create_KDTree():
    points = space_to_points()
    ab = [-55.45, 35.10]
    print(points[spatial.KDTree(points).query(ab)[1]])


def gaussian(v1, v2, sigma, sym=True):
    sig2 = 2 * sigma * sigma
    w = np.exp((v1-v2) ** 2 / sig2)
    return w


def nearest_quantized(ab):
    return


def define_in_gamut(ab_vals):
    space = np.zeros((22, 22))
    q = 0
    for ab in ab_vals:
        a, b = ab_to_quantization(ab)
        if not space[a, b]:
            space[a, b] = 1
            q += 1

    return space, q


def create_q_list():
    space = np.load('./quantized_space/q_space.npy')
    q_list = []
    for a in range(space.shape[0]):
        for b in range(space.shape[1]):
            if space[a, b]:
                q_list.append((a, b))
    print(len(q_list))
    np.save('./quantized_space/q_list.npy', np.array(q_list))


if __name__ == '__main__':
    # create_KDTree()
    # create_q_list()

    print((1, 2) == [1, 2])

    # ab_vals = torch.load('./processed_data/ab_values5000.pt')
    # space, q = define_in_gamut(ab_vals)

    # np.save('./quantized_space/q_space.npy', space)

    # plt.save('/graphs/gamut2.png')
    # plt.imshow(space)
    # plt.show()
    # print(space)
    # print('Q should be', q)
