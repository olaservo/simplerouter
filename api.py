from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
import json

DEFAULT_SYSTEM_MESSAGE = "You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest. Your goal is to provide informative and substantive responses to queries while avoiding potential harms."

NORDSTROM_TONE_MESSAGE = ""
try:
    with open('nordstrom_tone.md', 'r') as file:
        NORDSTROM_TONE_MESSAGE = file.read()
except:
    print("Error loading nordstrom_tone.md")

DEFAULT_REQUEST_PAYLOAD = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 250,
    "stop_sequences": ["\n\nHuman:"],
    "anthropic_version": "bedrock-2023-05-31",
    "system": DEFAULT_SYSTEM_MESSAGE
}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Initialize Bedrock client
session = boto3.Session(profile_name='nordstrom-federated')
bedrock = session.client(
    service_name='bedrock',
    region_name='us-west-2'
)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json

    # Extract parameters from request
    model = data.get('model', 'anthropic.claude-3-5-sonnet')
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 4096)
    temperature = data.get('temperature', 0.0)

    # Map model to Bedrock model ID
    bedrock_model_id = map_to_bedrock_model(model)

    request_body = DEFAULT_REQUEST_PAYLOAD
    # TODO: finalize request format from chat craft
    request_body['messages'] = messages
    request_body['max_tokens'] = int(max_tokens)
    request_body['temperature'] = float(temperature)

    try:
        # Call Bedrock API
        response = bedrock.invoke_model(
            modelId=bedrock_model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps(request_body)
        )

        # Process and return response
        result = json.loads(response['body'].read())

        # TODO: finalize response format
        # return jsonify({
        #     "id": "chatcmpl-" + str(uuid.uuid4()),
        #     "object": "chat.completion",
        #     "created": int(time.time()),
        #     "model": model,
        #     "choices": [{
        #         "index": 0,
        #         "message": {
        #             "role": "assistant",
        #             "content": result['completion']
        #         },
        #         "finish_reason": "stop"
        #     }]
        # })
        return jsonify(response['content'])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    # TODO: tweak response to match what chatcraft needs
    # also check limitations:
    # Some fields (like tokenizer and instruct_type) are not available from Bedrock API, so they're set to default values.
    # The modality is assumed to be "text->text" for all models, which may not be accurate for all Bedrock models.
    # Pricing information is included, but Bedrock's pricing structure might differ from OpenRouter's.
    # The is_moderated flag is set to False by default, as Bedrock doesn't provide this information.

    # To improve this further, you might need:

    # Documentation on Bedrock's model capabilities to accurately fill in fields like modality and tokenizer.
    # Information on Bedrock's pricing structure to ensure the pricing data is accurate and complete.
    # Details on any moderation or request limits Bedrock might impose.
    # Example of response from open router:
    # {
    #   "id": "anthropic/claude-3.5-sonnet",
    #   "name": "Anthropic: Claude 3.5 Sonnet",
    #   "created": 1718841600,
    #   "description": "Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Autonomously writes, edits, and runs code with reasoning and troubleshooting\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\n#multimodal",
    #   "context_length": 200000,
    #   "architecture": {
    #     "modality": "text+image-\u003Etext",
    #     "tokenizer": "Claude",
    #     "instruct_type": null
    #   },
    #   "pricing": {
    #     "prompt": "0.000003",
    #     "completion": "0.000015",
    #     "image": "0.0048",
    #     "request": "0"
    #   },
    #   "top_provider": {
    #     "context_length": 200000,
    #     "max_completion_tokens": 8192,
    #     "is_moderated": true
    #   },
    #   "per_request_limits": null
    # }
    

    # is moderated: Whether content filtering is applied by OpenRouter, per the model provider's Terms of Service.
    # Developers should adhere to the terms of the model regardless.
    # see: https://openrouter.ai/docs/models
    
    response = bedrock.list_foundation_models()
    models = []
    # Commenting out keys that don't exist in the provided response structure
    for model in response['modelSummaries']:
        input_modalities = '+'.join(model['inputModalities'])
        output_modalities = '+'.join(model['outputModalities'])
        modality = f"{input_modalities}->{output_modalities}".lower()
        model_data = {
            "id": model['modelId'],
            "name": model['modelName'],
            # "created": created_at,  # No 'createdAt' key in the provided structure
            # "description": model_info['modelDetails'].get('description', ''),  # No 'description' key in the provided structure
            "context_length": model.get('maximumInputTokenCount', 0),
            "architecture": {
                "modality": modality,
                "tokenizer": "Unknown",  # Assuming default value
                "instruct_type": None  # Assuming default value
            },
            "pricing": {
                "prompt": model.get('inputTokenPricePerUnit', 0),
                "completion": model.get('outputTokenPricePerUnit', 0),
                "image": 0,  # Assuming default value
                "request": 0  # Assuming default value
            },
            "top_provider": {
                "context_length": model.get('maximumInputTokenCount', 0),
                "max_completion_tokens": model.get('maximumOutputTokenCount', None),
                "is_moderated": False,  # Assuming default value
            },
            "per_request_limits": None  # Assuming default value
        }
        models.append(model_data)

    return jsonify({"data": models})


def map_to_bedrock_model(model):
    # Map OpenRouter-style model names to Bedrock model IDs
    model_map = {
        "anthropic.claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        # Add more mappings as needed
    }
    return model_map.get(model, model)

if __name__ == '__main__':
    app.run(debug=True)