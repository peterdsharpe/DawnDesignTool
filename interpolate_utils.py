import aerosandbox.numpy as np

def bridge_nans(array):  # TODO remove modification on incoming values; only patch nans
    """
    Patches NaN values in a 2D array. Can patch holes or entire regions. Uses Laplacian smoothing.
    :param array:
    :return:
    """
    original_nans = np.isnan(array)

    nanfrac = lambda array: np.sum(np.isnan(array)) / len(array.flatten())


    def item(i, j):
        if i < 0 or j < 0:  # don't allow wrapping other than what's controlled here
            return np.nan
        try:
            return array[i, j % array.shape[1]]  # allow wrapping around day of year
        except IndexError:
            return np.nan


    print_title = lambda name: print(f"{name}\nIter | NaN Fraction")
    print_progress = lambda iter: print(f"{iter:4} | {nanfrac(array):.6f}")

    # Bridging
    print_title("Bridging")
    print_progress(0)
    iter = 1
    last_nanfrac = nanfrac(array)
    making_progress = True
    while making_progress:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if not np.isnan(array[i, j]):
                    continue

                pairs = [
                    [item(i, j - 1), item(i, j + 1)],
                    [item(i - 1, j), item(i + 1, j)],
                    [item(i - 1, j + 1), item(i + 1, j - 1)],
                    [item(i - 1, j - 1), item(i + 1, j + 1)],
                ]

                for pair in pairs:
                    a = pair[0]
                    b = pair[1]

                    if not (np.isnan(a) or np.isnan(b)):
                        array[i, j] = (a + b) / 2
                        continue
        print_progress(iter)
        making_progress = nanfrac(array) != last_nanfrac
        last_nanfrac = nanfrac(array)
        iter += 1
