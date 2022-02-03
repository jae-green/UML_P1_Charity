# TODO: Fill in code in the function below to implement a gradient descent
# step for linear regression, following a squared error rule. See the docstring
# for parameters and returned variables.

def MSEStep(x, y, w_one, w_two, learn_rate = 0.005):
    """
    This function implements the gradient descent step for squared error as a
    performance metric.

    Parameters
    x : array of predictor features (x values of points)
    y : array of outcome values (y values of points)
    w_one : predictor feature coefficients (aka slope of line)
    w_two : regression function intercept (y-intercept of line)
    learn_rate : learning rate (alpha value)

    Returns
    w_one_new : predictor feature coefficients following gradient descent step
    w_two_new : intercept following gradient descent step
    """

    # Determine the value of error
    y_hat = np.matmul(x, w_one) + w_two
    error = y - y_hat

    # Complete "Square Trick" on each based on above error
    w1_new = w_one + learn_rate * np.matmul(error, x)
    w2_new = w_two + learn_rate * error.sum()

    return w1_new, w2_new
