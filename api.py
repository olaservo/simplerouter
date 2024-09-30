import uuid
import time
import json
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import boto3

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

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': 'http://localhost:5173'}})

# Initialize Bedrock client
session = boto3.Session(profile_name='nordstrom-federated')

bedrock_runtime = session.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'
)

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    model = data.get('model', DEFAULT_MODEL_ID)
    max_tokens = data.get('max_tokens', DEFAULT_REQUEST_PAYLOAD['max_tokens'])
    temperature = data.get('temperature', DEFAULT_REQUEST_PAYLOAD['temperature'])
    request_messages = data.get('messages', [])
    stream = data.get('stream', False)

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

    try:
        if stream:
            response = bedrock_runtime.converse_stream(
                modelId=model,
                messages=messages,
                inferenceConfig={
                    'maxTokens': int(max_tokens),
                    'temperature': float(temperature),
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
                    'maxTokens': int(max_tokens),
                    'temperature': float(temperature),
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
    # TODO: tweak response to match what chatcraft needs
    # To improve this further, you might need:
    # Documentation on Bedrock's model capabilities to accurately fill in fields like modality and tokenizer.
    # Information on Bedrock's pricing structure to ensure the pricing data is accurate and complete.
    # Details on any moderation or request limits Bedrock might impose.
    # Example of response from open router:
    # {
    #   'id': 'anthropic/claude-3.5-sonnet',
    #   'name': 'Anthropic: Claude 3.5 Sonnet',
    #   'created': 1718841600,
    #   'description': 'Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Autonomously writes, edits, and runs code with reasoning and troubleshooting\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\n#multimodal',
    #   'context_length': 200000,
    #   'architecture': {
    #     'modality': 'text+image-\u003Etext',
    #     'tokenizer': 'Claude',
    #     'instruct_type': null
    #   },
    #   'pricing': {
    #     'prompt': '0.000003',
    #     'completion': '0.000015',
    #     'image': '0.0048',
    #     'request': '0'
    #   },
    #   'top_provider': {
    #     'context_length': 200000,
    #     'max_completion_tokens': 8192,
    #     'is_moderated': true
    #   },
    #   'per_request_limits': null
    # }
    
    # TODO: add option to support provisioned
    response = bedrock.list_foundation_models(byInferenceType=DEFAULT_PRICING_STRUCTURE)
    models = []
    for model in response['modelSummaries']:
         # TODO: add support for other models
        if not model['modelId'].startswith('anthropic'):
            continue
        input_modalities = '+'.join(model['inputModalities'])
        output_modalities = '+'.join(model['outputModalities'])
        modality = f'{input_modalities}->{output_modalities}'.lower()
        model_data = {
            'id': model['modelId'],
            'name': model['modelName'],
            # 'created': created_at,  # No 'createdAt' key in the provided structure #TODO
            # 'description': model_info['modelDetails'].get('description', ''),  # No 'description' key in the provided structure #TODO
            'context_length': model.get('maximumInputTokenCount', 0), #TODO
            'architecture': {
                'modality': modality,
                'tokenizer': 'Unknown',  #TODO
                'instruct_type': None  #TODO
            },
            'pricing': {
                'prompt': model.get('inputTokenPricePerUnit', 0), #TODO
                'completion': model.get('outputTokenPricePerUnit', 0), #TODO
                'image': 0,  #TODO
                'request': 0  #TODO
            },
            'top_provider': {
                'context_length': model.get('maximumInputTokenCount', 0), #TODO
                'max_completion_tokens': model.get('maximumOutputTokenCount', None), #TODO
                'is_moderated': False,  #TODO
            },
            'per_request_limits': None  #TODO
        }
        models.append(model_data)

    return jsonify({'data': models})


if __name__ == '__main__':
    app.run(debug=True)