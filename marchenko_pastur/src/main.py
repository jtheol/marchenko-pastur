import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def create_dataset(num_samples: int = 50, num_features: int = 500) -> np.ndarray:
    """
    Generates a random dataset with a specified number of samples and features, computes
    the covariance matrix of the dataset, and returns the eigenvalues of the covariance matrix.

    Args:
        num_samples (int): The number of observations (samples) to generate. Defaults to 50.
        num_features (int): The number of variables (features) to generate for each sample. Defaults to 500.

    Returns:
        np.ndarray: An array of eigenvalues obtained from the covariance matrix of the
                    generated dataset.
    """

    data = np.random.normal(size=(num_samples, num_features))
    cov_matrix = np.cov(data, rowvar=False)
    eigen_vals = np.linalg.eigvals(cov_matrix)

    return eigen_vals


def marchenko_pastur_pdf(n: int, q: float, sigma: float = 1.0) -> tuple:
    """
    Computes the Marchenko–Pastur probability density function.

    Args:
        n (int): The number of points to generate.
        q (float): The aspect ratio, defined as the ratio of the number of features to the number of samples (q = num_features / num_samples).
        sigma (float): The standard deviation. Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - pdf (np.ndarray): The PDF values of the Marchenko–Pastur distribution.
            - data (np.ndarray): The corresponding eigenvalue range (data) over which the PDF is evaluated.
            - lambda_minus (float): The lower bound of the Marchenko–Pastur distribution (lambda_minus).
            - lambda_plus (float): The upper bound of the Marchenko–Pastur distribution (lambda_plus).
    """

    lambda_minus = sigma**2 * (1 - np.sqrt(1 / q)) ** 2
    lambda_plus = sigma**2 * (1 + np.sqrt(1 / q)) ** 2

    data = np.linspace(lambda_minus, lambda_plus, n)

    pdf = (
        q
        * np.sqrt((lambda_plus - data) * (data - lambda_minus))
        / (2 * np.pi * data * sigma**2)
    )

    return pdf, data, lambda_minus, lambda_plus


if __name__ == "__main__":
    num_samples = [50, 150, 250, 300, 500]
    num_features = 500

    significant_vals = []

    for num_sample in num_samples:
        q = num_features / num_sample

        eigen_vals = create_dataset(num_sample, num_features)

        mp_pdf, data, lm, lp = marchenko_pastur_pdf(num_features, q)

        significant_eigenvalues = eigen_vals[(eigen_vals < lm) | (eigen_vals > lp)]

        significant_vals.append(
            {
                "num_samples": num_sample,
                "q": q,
                "lm": lm,
                "lp": lp,
                "significant": significant_eigenvalues.shape[0],
            }
        )

        plt.hist(eigen_vals, bins=30, density=True, alpha=0.8)
        plt.plot(data, mp_pdf, label="Marchenko-Pastur distribution", color="red")
        plt.xlabel("Eigenvalues")
        plt.ylabel("Density")
        plt.title("Eigenvalue Distribution vs Marchenko-Pastur")
        plt.legend()
        plt.show()

    pprint(significant_vals)
