from flask import Flask, request, jsonify
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

# Initialize Bedrock client
session = boto3.Session(profile_name='nordstrom-federated')
bedrock = session.client(
    service_name='bedrock-runtime',
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