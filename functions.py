import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

#q = 246  # This was the Q we got, right?
q_space = np.load('./quantized_space/q_space.npy')
w_points = torch.from_numpy(np.load('./quantized_space/w_points.npy'))


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
    print(q_dist_img)
    q_dist_img = q_dist_img.T
    h, w, q = q_dist_img.shape
    ab_img = np.zeros((h, w, 2))

    for i in range(h):
        for j in range(w):
            q_dist_img[i, j] += w_points
            q_idx = torch.argmax(q_dist_img[i, j])
            a, b = q_points[q_idx]
            # print(q_points)
            # a, b = quantization_to_ab(q_val)
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


def one_hot_q(ab_img, load_data=False, q=246):
    # Need to turn matrix with a values and matrix with b values into list of values(a,b)
    if not load_data:
        ab_img = ab_img.T
    h, w, _ = ab_img.shape

    ab_one_hot = np.zeros((h, w, q))
    for i in range(h):
        for j in range(w):
            # a, b = ab_img[i, j]
            if not load_data:
                closest_p, idx, dist = find_k_nearest_q(ab_img[i, j].cpu())
            else:
                closest_p, idx, dist = find_k_nearest_q(ab_img[i, j])
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
            points.append([i*10-110, j*10-110])
    np.save('./quantized_space/q_points.npy', points)
    return points


def x_space_to_x_points(space, name):
    points = []
    for i in range(q_space.shape[0]):
        for j in range(q_space.shape[1]):
            if q_space[i, j] != 0:
                points.append(space[i, j])
    np.save('./quantized_space/'+name+'.npy', points)
    return np.array(points)


def find_k_nearest_q(ab, k=1):
    """
    INPUT:  The point ab. Wants point format [a,b] for ab.
            Optionally k, the number of points requested. Default 1.
    OUTPUT: The k closest point(s)
            their indexes in the list of q
            and the distance(s) to them
    """
    # print(ab)
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

def get_ab_pairs_quantified():
    ab_vals = torch.load('./processed_data/ab_values5000.pt')
    ab_pairs = np.zeros((22, 22))
    for ab in ab_vals:
        ab_pairs[ab_to_quantization(ab)] += 1
    np.save('./quantized_space/ab_pairs.npy', ab_pairs)
    return ab_pairs


def create_w_points():
    _lambda, _q = 0.5, 208
    ab_pairs = np.load('./quantized_space/ab_pairs.npy')
    print('ab pairs', ab_pairs)
    p = x_space_to_x_points(ab_pairs, 'p_points')
    print('p_points', p)
    #p = softmax(p)       # Try skipping first softmax ?
    #print('P2', p)
    denom = (1-_lambda) * p + (_lambda / _q)
    w = 1 / denom
    print('W1', w)
    w = softmax(w)
    print('W2', w)
    np.save('./quantized_space/p_points.npy', p)
    np.save('./quantized_space/w_points.npy', w)
    return w


def softmax(x):
    # save typing...
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    print(define_in_gamut(torch.load('./processed_data/ab_values5000.pt')))
    get_ab_pairs_quantified()
    create_w_points()
    # w_space_to_w_points()
    # ab_vals = np.load('./processed_data/ab_values50000000.npy')
    # q_list = np.load('./quantized_space/q_list.npy')

    # q_distrib_list = np.zeros((208, 1))
    # print(q_list.shape)

    # for ab in ab_vals:
    #     a, b = ab
    #     nearest, idx, dist = find_k_nearest_q([a, b])
    #     # print(idx)
    #     q_distrib_list[idx] += 1

    # print(q_distrib_list)

    """ space = np.load('./quantized_space/q_space.npy')
    plt.imshow(space)
    plt.show() """
