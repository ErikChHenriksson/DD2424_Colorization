import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

q = 208  # This was the Q we got, right?
q_space = np.load('./quantized_space/q_space.npy')


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


def quantization_to_ab(q_val):
    aq, bq = q_val
    a = math.floor((aq * 10)-110)
    b = math.floor((bq * 10)-110)
    return a, b


def q_distribution_to_ab(q_dist_img, q_points):
    q_dist_img = q_dist_img.T
    h, w, q = q_dist_img.shape
    ab_img = np.zeros((h, w, 2))

    for i in range(h):
        for j in range(w):
            print(q_dist_img[i, j])
            q_idx = torch.argmax(q_dist_img[i, j])
            a, b = q_points[q_idx]
            a -= 110
            b -= 110
            # a, b = quantization_to_ab(q_val)
            print('ab: ', (a, b))
            ab_img[i, j, :] = [a, b]

    return ab_img


def one_hot_quantization(ab):
    h, w, _ = ab.shape
    ab_one_hot = np.zeros((h, w, 22, 22))
    for i in range(h):
        for j in range(w):
            one_hot = np.zeros((22, 22))
            aq, bq = ab_to_quantization((i, j))
            one_hot[aq, bq] = 1
            ab_one_hot[i, j, :, :] = one_hot

    return ab_one_hot


def one_hot_q(ab_img, q_list):
    ab_img = ab_img.T
    h, w, _ = ab_img.shape
    q = q_list.shape[0]

    ab_one_hot = np.zeros((h, w, q))

    for i in range(h):
        for j in range(w):
            # a, b = ab_img[i, j]
            closest_p, idx, dist = find_k_nearest_q(ab_img[i, j].cpu())
            # x = np.where(q_list[:, 0] == a)
            # y = np.where(q_list[:, 1] == b)
            # idx = np.intersect1d(x, y)
            ab_one_hot[i, j, idx] = 1

    return ab_one_hot


def space_to_points():
    # Saves/returns the gamut as a list of points
    points = []
    for i in range(q_space.shape[0]):
        for j in range(q_space.shape[1]):
            if q_space[i, j] == 0:
                continue
            points.append([q_space[i, j]*i*10+5, q_space[i, j]*j*10+5])
    np.save('./quantized_space/q_points.npy', points)
    return points


def find_k_nearest_q(ab, k=1):
    """ 
    INPUT:  The point ab. Wants point format [a,b] for ab.
            Optionally k, the number of points requested. Default 1.
    OUTPUT: The k closest point(s)
            and the distance(s) to them
    """
    points = np.load('./quantized_space/q_points.npy')
    # plt.plot(np.array(points)[:, 0], np.array(points)[:, 1])
    # plt.show()
    tree = KDTree(points)
    dist = tree.query(ab, k=k)[0]
    index = tree.query(ab, k=k)[1]
    closest_point = points[index]
    return closest_point, index, dist


def one_hot_nearest_q(ab):
    nearest_q, _, _ = find_k_nearest_q(ab)
    print(nearest_q)
    return one_hot_quantization(nearest_q)


def gaussian(v1, v2, sigma):
    sig2 = 2 * sigma * sigma
    w = np.exp((v1-v2) ** 2 / sig2)
    return w


def i_to_q():
    return


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
    np.save('./quantized_space/q_space.npy', space)
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
    nearest = one_hot_nearest_q([60, 130])
    print(nearest)

    # q_list = np.load('./quantized_space/q_list.npy')
    # print(q_list[15])
    # x = np.where(q_list[:, 0] == 6)
    # y = np.where(q_list[:, 1] == 10)
    # print(np.intersect1d(x, y))
    # print(x, y)

    a = torch.randn(100, 32, 32, 208)
    sum = torch.sum(torch.sum(torch.sum(a, dim=3), dim=1), dim=1)
    print(sum)
    print(sum.shape)

    # plt.save('/graphs/gamut2.png')
    # plt.imshow(space)
    # plt.show()
    # print(space)
    # print('Q should be', q)
