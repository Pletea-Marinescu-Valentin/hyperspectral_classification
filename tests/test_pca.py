import unittest
import numpy as np
from preprocessing.pca import PCA

class TestPCA(unittest.TestCase):
    def test_pca_transformation(self):
        # Sample data: 4 samples with 3 features each
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])

        # Initialize PCA with 2 components
        pca = PCA(n_components=2, random_state=42)
        transformed_data = pca.fit_transform(data)

        # Check the shape of the transformed data
        self.assertEqual(transformed_data.shape, (4, 2))

if __name__ == "__main__":
    unittest.main()