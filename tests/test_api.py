import json
import pytest
from flask import url_for
from io import BytesIO
from unittest.mock import patch, MagicMock
from simplerouter.api import app, bedrock_runtime

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_bedrock_runtime():
    with patch('simplerouter.api.bedrock_runtime') as mock_client:
        yield mock_client

def load_mock_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

@pytest.fixture
def mock_list_models_response():
    return load_mock_data('tests/mocks/aws_model_summaries_mock.json')

@pytest.fixture
def mock_invoke_model_response():
    return {
        'output': {
            'message': {
                'role': 'assistant',
                'content': [{'text': 'This is a mocked response from the Bedrock API.'}]
            }
        },
        'usage': {
            'inputTokens': 10,
            'outputTokens': 20
        },
        'stopReason': 'COMPLETE'
    }

def test_chat_completion_without_file(client, mock_bedrock_runtime, mock_invoke_model_response):
    mock_bedrock_runtime.converse.return_value = mock_invoke_model_response
    
    data = {
        'model': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
        'max_tokens': 100,
        'temperature': 0.7,
        'messages': json.dumps([
            {"role": "user", "content": "Hello, how are you?"}
        ])
    }
    response = client.post('/chat/completions', data=data)
    assert response.status_code == 200
    result = json.loads(response.data)
    assert 'choices' in result
    assert len(result['choices']) > 0
    assert 'message' in result['choices'][0]
    assert 'content' in result['choices'][0]['message']
    assert 'mocked response' in result['choices'][0]['message']['content']

def test_chat_completion_with_file(client, mock_bedrock_runtime, mock_invoke_model_response):
    mock_bedrock_runtime.converse.return_value = mock_invoke_model_response
    
    data = {
        'model': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
        'max_tokens': 100,
        'temperature': 0.7,
        'messages': json.dumps([
            {"role": "user", "content": "Please analyze the content of the uploaded file."}
        ])
    }
    file_content = b"This is a test file content."
    data['file'] = (BytesIO(file_content), 'test.txt')
    
    response = client.post('/chat/completions', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert 'choices' in result
    assert len(result['choices']) > 0
    assert 'message' in result['choices'][0]
    assert 'content' in result['choices'][0]['message']
    assert 'mocked response' in result['choices'][0]['message']['content']

def test_chat_completion_error_handling(client, mock_bedrock_runtime):
    mock_bedrock_runtime.converse.side_effect = Exception("Invalid model")
    
    data = {
        'model': 'invalid_model',
        'max_tokens': 100,
        'temperature': 0.7,
        'messages': json.dumps([
            {"role": "user", "content": "Hello, how are you?"}
        ])
    }
    response = client.post('/chat/completions', data=data)
    assert response.status_code == 500
    result = json.loads(response.data)
    assert 'error' in result

@patch('simplerouter.api.bedrock')
def test_list_models(mock_bedrock, client, mock_list_models_response):
    mock_bedrock.list_foundation_models.return_value = mock_list_models_response
    
    response = client.get('/models')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert 'data' in result
    assert len(result['data']) > 0
    assert 'id' in result['data'][0]
    assert 'name' in result['data'][0]
