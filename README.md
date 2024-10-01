# simplerouter

SimpleRouter is a Flask app to let you chat with GenAI models locally and/or from your own cloud accounts.

Primary motivation is to avoid sharing data to a third-party service while allowing users to easily experiment with a variety of GenAI models.

It is intended to be used together with [ChatCraft](https://github.com/tarasglek/chatcraft.org) as a client, but it can be used with any client that can send and receive JSON messages over a WebSocket connection.

## Supported cloud providers and models

* AWS Bedrock
  * [Anthropic models](https://aws.amazon.com/bedrock/claude/) (On-Demand Only)

Note that not all features are enabled yet, including specifying content filters, tracing, etc.

## Setup steps

### One-time setup for Flask app
1. Clone this repository
2. Create virtual environment and install dependencies:
   1. `python3 -m venv venv`
   2. `source venv/bin/activate`
   3. `pip install -r requirements.txt`
3. Set up .env file with the following variables: Set the profile name in .env file #TODO test this and see if you need it

### Running the Flask app
1. Log into aws (details not included here) and select sandbox role #TODO
2. Start the Flask app: #TODO test this
   1. `flask run`

### Setup for ChatCraft UI

Refer to the [ChatCraft README](https://github.com/tarasglek/chatcraft.org) for instructions on how to set up the ChatCraft UI.

Once you have the UI running, go to Settings and add the following provider details:

* Name: Localhost
* Host: http://127.0.0.1:5000/
* API key: dummy_key (this can be anything that isn't blank)

Then hit Save.  Now you should be able to select Localhost as a provider in the chat UI.  If you see any errors, take a look at the terminal output for the simplerouter Flask app.

Note that CORS is configured to allow requests from http://localhost:5173 which is the default port for ChatCraft.  If you run this from a different port, you will need to update the CORS configuration in `api.py`.

