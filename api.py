import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import boto3

load_dotenv()

app = Flask(__name__)

# Initialize Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json

    # Extract parameters from request
    model = data.get('model', 'anthropic.claude-v2')  # Default to Claude v2
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)

    # Map model to Bedrock model ID
    bedrock_model_id = map_to_bedrock_model(model)

    # Prepare input for Bedrock
    prompt = prepare_prompt(messages)

    try:
        # Call Bedrock API
        response = bedrock.invoke_model(
            modelId=bedrock_model_id,
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature
            })
        )

        # Process and return response
        result = json.loads(response['body'].read())
        return jsonify({
            "id": "chatcmpl-" + str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result['completion']
                },
                "finish_reason": "stop"
            }]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def map_to_bedrock_model(model):
    # Map OpenRouter-style model names to Bedrock model IDs
    model_map = {
        "anthropic.claude-v2": "anthropic.claude-v2",
        "anthropic.claude-instant-v1": "anthropic.claude-instant-v1",
        # Add more mappings as needed
    }
    return model_map.get(model, model)

def prepare_prompt(messages):
    # Convert messages to Anthropic-style prompt
    prompt = ""
    for message in messages:
        role = message['role']
        content = message['content']
        if role == 'system':
            prompt += f"Human: {content}\n\nAssistant: Understood. How can I assist you?\n\n"
        elif role == 'user':
            prompt += f"Human: {content}\n\n"
        elif role == 'assistant':
            prompt += f"Assistant: {content}\n\n"
    prompt += "Assistant:"
    return prompt

if __name__ == '__main__':
    app.run(debug=True)