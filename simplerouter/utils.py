import json
import os
from decimal import Decimal, ROUND_HALF_UP

# Load model details from JSON file
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'provider_model_details.json')
with open(json_path, 'r') as f:
    model_details = json.load(f)['data']

def calculate_costs(model_id, input_tokens, output_tokens):
    model = next((m for m in model_details if m['id'] == model_id), None)
    if not model:
        return None

    pricing = model['pricing']
    input_cost = (Decimal(input_tokens) / 1000) * Decimal(str(pricing['prompt']))
    output_cost = (Decimal(output_tokens) / 1000) * Decimal(str(pricing['completion']))
    total_cost = input_cost + output_cost

    return {
        'input_cost': float(input_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)),
        'output_cost': float(output_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)),
        'total_cost': float(total_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))
    }
