import unittest
from unittest.mock import patch, mock_open
import json
from simplerouter.utils import calculate_costs

class TestUtils(unittest.TestCase):

    @patch('simplerouter.utils.open', new_callable=mock_open, read_data=json.dumps({
        "data": [
            {
                "id": "openai.gpt-3.5-turbo",
                "pricing": {
                    "prompt": 0.0005,
                    "completion": 0.0015
                }
            }
        ]
    }))
    def test_calculate_costs(self, mock_file):
        # Test with valid inputs
        result = calculate_costs("openai.gpt-3.5-turbo", 1000, 500)
        self.assertAlmostEqual(result['input_cost'], 0.0005, places=6)
        self.assertAlmostEqual(result['output_cost'], 0.00075, places=6)
        self.assertAlmostEqual(result['total_cost'], 0.00125, places=6)

        # Test with zero tokens
        result = calculate_costs("openai.gpt-3.5-turbo", 0, 0)
        self.assertEqual(result, {
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0
        })

        # Test with non-existent model
        result = calculate_costs("non-existent-model", 1000, 500)
        self.assertIsNone(result)

    @patch('simplerouter.utils.open', new_callable=mock_open, read_data=json.dumps({
        "data": [
            {
                "id": "openai.gpt-3.5-turbo",
                "pricing": {
                    "prompt": 0.0005,
                    "completion": 0.0015
                }
            }
        ]
    }))
    def test_calculate_costs_rounding(self, mock_file):
        # Test rounding to 6 decimal places
        result = calculate_costs("openai.gpt-3.5-turbo", 1234567, 7654321)
        self.assertAlmostEqual(result['input_cost'], 0.617284, places=6)
        self.assertAlmostEqual(result['output_cost'], 11.481482, places=6)
        self.assertAlmostEqual(result['total_cost'], 12.098765, places=6)

if __name__ == '__main__':
    unittest.main()
