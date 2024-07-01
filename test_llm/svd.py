import numpy as np


def svd(X, r=2):
    # Data matrix X, X doesn't need to be 0-centered
    n, m = X.shape
    # Compute full SVD
    U, Sigma, Vh = np.linalg.svd(
        X,
        full_matrices=False,  # It's not necessary to compute the full matrix of U or V
        compute_uv=True,
    )
    # Transform X with SVD components
    U = U[:, :r]
    Sigma = Sigma[:r]
    Vh = Vh[:r, :]
    X_svd = np.dot(np.dot(U, np.diag(Sigma)), Vh)
    return X_svd


def f_norm(x, y):
    sub_ = x - y
    return np.linalg.norm(sub_)


if __name__ == "__main__":
    # init_x = [[5, 6, 7], [1, 2, 3], [4, 5, 8], [8, 11, 2], [3, 7, 9]]
    x_1 = np.array([[5, 6, 7], [1, 2, 3]])
    x_2 = np.array([[5, 6, 7], [1, 2.1, 3.1]])

    init_x = np.random.rand(100, 2000)
    v_split_line = 50
    h_split_line = 100
    x = np.array(init_x)
    # up_down
    x_up = init_x[:v_split_line]
    x_down = init_x[v_split_line:]
    r = 10
    tz_r = 0
    svd_x = svd(x, r)
    svd_x_up = svd(x_up, r)
    svd_x_down = svd(x_down, r)
    svd_new = np.vstack((svd_x_up, svd_x_down))
    # print(x, svd_x, f_norm(x, svd_x), f_norm(x_1, x_2), sep="\n")
    # print(f"origin x: \n{x}")
    # print(f"svd x: \n{svd_x}")
    print(f"f norm of the svd x: \n{f_norm(x, svd_x)}")
    # print(f"svd_combie x: \n{svd_new}")
    print(f"f norm of the svd_combie_2_pw x: \n{f_norm(x, svd_new)}")
    # print(f"origin x: {x}")
    # left_right
    x_left = init_x[:, :v_split_line]
    x_right = init_x[:, v_split_line:]
    svd_x_left = svd(x_left, r)
    svd_x_right = svd(x_right, r)
    svd_new_lr = np.hstack((svd_x_left, svd_x_right))
    # print(x, svd_x, f_norm(x, svd_x), f_norm(x_1, x_2), sep="\n")
    # print(f"origin x: \n{x}")
    # print(f"svd x: \n{svd_x}")
    # print(f"f norm of the svd x: \n{f_norm(x, svd_x)}")
    # print(f"svd_combie x: \n{svd_new}")
    print(f"f norm of the svd_combie_2_lr x: \n{f_norm(x, svd_new_lr)}")
    x_up_left = init_x[:v_split_line, :h_split_line]
    x_up_right = init_x[:v_split_line, h_split_line:]
    x_down_left = init_x[v_split_line:, :h_split_line]
    x_down_right = init_x[v_split_line:, h_split_line:]
    svd_x_up_left = svd(x_up_left, r + tz_r)
    svd_x_up_right = svd(x_up_right, r - tz_r)
    svd_x_down_left = svd(x_down_left, r - tz_r)
    svd_x_down_right = svd(x_down_right, r + tz_r)
    svd_new_up = np.hstack((svd_x_up_left, svd_x_up_right))
    svd_new_down = np.hstack((svd_x_down_left, svd_x_down_right))
    svd_new_4 = np.vstack((svd_new_up, svd_new_down))
    print(f"f norm of the svd_combie_4 x: \n{f_norm(x, svd_new_4)}\n\n")
