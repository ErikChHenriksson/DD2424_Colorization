def getQ(pixels):
    colors = np.zeros((22, 22))

    for p in pixels:
        a, b = p
        colors[get_index(a), get_index(b)] = 1

    return np.count_nonzero(colors)


def get_index(num):
    (num + 110) / 10
