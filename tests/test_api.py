import json
import pytest
from flask import url_for
from io import BytesIO
from simplerouter.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_chat_completion_without_file(client):
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

def test_chat_completion_with_file(client):
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
    assert "test file content" in result['choices'][0]['message']['content'].lower()

def test_chat_completion_error_handling(client):
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

def test_list_models(client):
    response = client.get('/models')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert 'data' in result
    assert len(result['data']) > 0
    assert 'id' in result['data'][0]
    assert 'name' in result['data'][0]
