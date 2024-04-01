def mse_loss(ground_truth, output):
    squared_diff = (ground_truth - output) ** 2
    overall_mse = squared_diff.mean()
    return overall_mse
