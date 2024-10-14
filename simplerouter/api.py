import uuid
import time
import json
import os
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import boto3
from werkzeug.utils import secure_filename

DEFAULT_MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

DEFAULT_PRICING_STRUCTURE = 'ON_DEMAND'

DEFAULT_REQUEST_PAYLOAD = {
    'max_tokens': 4096,
    'temperature': 0.0,
    'top_p': 0.9,
    'top_k': 250,
    'stop_sequences': ['\n\nHuman:'],
    'anthropic_version': 'bedrock-2023-05-31',
}

# Load model details from JSON file
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'provider_model_details.json')
with open(json_path, 'r') as f:
    model_details = json.load(f)['data']

# Load allowed providers from config file
config_dir = os.path.join(os.path.dirname(current_dir), 'config')
allowed_providers_path = os.path.join(config_dir, 'allowed_providers.json')
with open(allowed_providers_path, 'r') as f:
    allowed_providers = json.load(f)['allowed_providers']

app = Flask(__name__)
# ChatCraft runs on port 5173 by default
CORS(app, resources={r'/*': {'origins': 'http://localhost:5173'}})

# Initialize Bedrock client
session = boto3.Session()
bedrock_runtime = session.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'
)

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    data = request.form.to_dict()
    model = data.get('model', DEFAULT_MODEL_ID)
    max_tokens = int(data.get('max_tokens', DEFAULT_REQUEST_PAYLOAD['max_tokens']))
    temperature = float(data.get('temperature', DEFAULT_REQUEST_PAYLOAD['temperature']))
    request_messages = json.loads(data.get('messages', '[]'))
    stream = data.get('stream', 'false').lower() == 'true'

    messages = []
    system = []
    for message in request_messages:
        if message['role'] == 'system':
            system = [{'text': message['content']}]
        else:
            messages.append({
                'role': message['role'],
                'content': [{'text': message['content']}]
            })

    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_content = file.read().decode('utf-8')
            messages.append({
                'role': 'user',
                'content': [
                    {'text': f"Here's the content of the file {filename}:\n\n{file_content}\n\nPlease analyze this file content."}
                ]
            })

    try:
        if stream:
            response = bedrock_runtime.converse_stream(
                modelId=model,
                messages=messages,
                inferenceConfig={
                    'maxTokens': max_tokens,
                    'temperature': temperature,
                    'topP': float(DEFAULT_REQUEST_PAYLOAD['top_p']),
                    'stopSequences': ['\n\nHuman:'],
                },
                system=system,
                additionalModelRequestFields={'top_k': int(DEFAULT_REQUEST_PAYLOAD['top_k'])}
            )
            return Response(stream_with_context(stream_response(response, model)), content_type='text/event-stream')
        else:
            response = bedrock_runtime.converse(
                modelId=model,
                messages=messages,
                inferenceConfig={
                    'maxTokens': max_tokens,
                    'temperature': temperature,
                    'topP': float(DEFAULT_REQUEST_PAYLOAD['top_p']),
                    'stopSequences': ['\n\nHuman:'],
                },
                system=system,
                additionalModelRequestFields={'top_k': int(DEFAULT_REQUEST_PAYLOAD['top_k'])}
            )
            return jsonify(process_non_stream_response(response, model))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def stream_response(response, model):
    for event in response['stream']:
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']
            if 'text' in delta:
                chunk = {
                    'id': f'chatcmpl-{str(uuid.uuid4())}',
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': model,
                    'choices': [{
                        'index': 0,
                        'delta': {
                            'content': delta['text']
                        },
                        'finish_reason': None
                    }]
                }
                yield f'data: {json.dumps(chunk)}\n\n'
        elif 'messageStop' in event:
            chunk = {
                'id': f'chatcmpl-{str(uuid.uuid4())}',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': model,
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': event['messageStop']['stopReason']
                }]
            }
            yield f'data: {json.dumps(chunk)}\n\n'
            yield 'data: [DONE]\n\n'


def process_non_stream_response(response, model):
    input_tokens = response['usage']['inputTokens']
    output_tokens = response['usage']['outputTokens']
    return {
        'id': 'chatcmpl-' + str(uuid.uuid4()),
        'created': int(time.time()),
        'model': model,
        'object': 'chat.completion',
        'choices': [{
            'message': {
                'role': response['output']['message']['role'],
                'content': response['output']['message']['content'][0]['text']
            },
            'finish_reason': response['stopReason']
        }],
        'usage': {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
        },
    }

bedrock = session.client(
    service_name='bedrock',
    region_name='us-west-2'
)

@app.route('/models', methods=['GET'])
def list_models():
    response = bedrock.list_foundation_models(byInferenceType=DEFAULT_PRICING_STRUCTURE)
    models = []
    for model in response['modelSummaries']:
        # Check if the model's provider is in the allowed_providers list
        if not any(model['modelId'].startswith(provider) for provider in allowed_providers):
            continue
        input_modalities = '+'.join(model['inputModalities'])
        output_modalities = '+'.join(model['outputModalities'])
        modality = f'{input_modalities}->{output_modalities}'.lower()

        # Find corresponding model details from JSON for anything we can't get from the list_foundation_models call yet
        model_detail = next((item for item in model_details if item['id'] == model['modelId']), {})

        # Translate to OpenRouter format
        model_data = {
             'id': model['modelId'],
             'name': model['modelName'],
             'context_length': model.get('maximumInputTokenCount', model_detail.get('context_length', 0)),
             'architecture': {
                 'modality': modality,
                 'tokenizer': model_detail.get('architecture', {}).get('tokenizer', 'Unknown'),
                 'instruct_type': model_detail.get('architecture', {}).get('instruct_type', None)
             },
             'pricing': {
                 'prompt': model.get('inputTokenPricePerUnit', model_detail.get('pricing', {}).get('prompt', 0)),
                 'completion': model.get('outputTokenPricePerUnit', model_detail.get('pricing', {}).get('completion', 0)),
                 'image': model_detail.get('pricing', {}).get('image', 0),
                 'request': model_detail.get('pricing', {}).get('request', 0)
             },
             'top_provider': {
                 'context_length': model.get('maximumInputTokenCount', model_detail.get('top_provider', {}).get('context_length', 0)),
                 'max_completion_tokens': model.get('maximumOutputTokenCount', model_detail.get('top_provider', {}).get('max_completion_tokens', None)),
                 'is_moderated': model_detail.get('top_provider', {}).get('is_moderated', False)
             },
             'per_request_limits': model_detail.get('per_request_limits', None)
         }
        models.append(model_data)

    return jsonify({'data': models})


if __name__ == '__main__':
    app.run(debug=True)
