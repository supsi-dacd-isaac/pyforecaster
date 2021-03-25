import unittest
import pandas as pd
import numpy as np
from pyforecaster.pyforecaster import format_dataset


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 100
        self.n = 10
        self.x = pd.DataFrame(np.random.randn(self.t, self.n))

    def test_returned_type(self):
        x_frmt = format_dataset(self.x)
        assert isinstance(x_frmt, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
