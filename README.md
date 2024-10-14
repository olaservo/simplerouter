# simplerouter

SimpleRouter is a Flask app to let you chat with GenAI models locally and/or from your own cloud accounts.

Primary motivation is to avoid sharing data to a third-party service while allowing users to easily experiment with a variety of GenAI models.

It is intended to be used together with [ChatCraft](https://github.com/tarasglek/chatcraft.org) as a client, but it can be used with any client that can send and receive JSON messages over a WebSocket connection.

## Supported cloud providers and models so far

* AWS Bedrock
  * [Anthropic models](https://aws.amazon.com/bedrock/claude/) (On-Demand Only)

Note that not all features are enabled yet, including specifying content filters, tracing, etc.

## Setup steps

### One-time setup for Flask app
1. Clone this repository
2. Install Poetry (if not already installed):
   ```
   pipx install poetry
   ```
3. Install project dependencies:
   ```
   poetry install
   ```
4. Set up .env file with the following variables: Set the profile name in .env file #TODO test this and see if you need it

### Running the Flask app
1. Log into aws (details not included here) and select sandbox role #TODO
2. Start the Flask app by running the following command in your terminal:
   ```
   poetry run flask --app simplerouter.api run
   ```

### Setup for ChatCraft UI

Refer to the [ChatCraft README](https://github.com/tarasglek/chatcraft.org) for instructions on how to set up the ChatCraft UI.

Once you have the UI running, go to Settings and add the following provider details:

* Name: Localhost
* Host: http://127.0.0.1:5000/
* API key: dummy_key (this can be anything that isn't blank)

Then hit Save.  Now you should be able to select Localhost as a provider in the chat UI.  If you see any errors, take a look at the terminal output for the simplerouter Flask app.

To see token count as you chat, go to Settings >> Custom Settings and enable "Track and Display Token Count and Cost".  Cost is not currently displayed for most models, but token count should be accurate and cost estimation is coming soon.

Note that CORS is configured to allow requests from http://localhost:5173 which is the default port for ChatCraft.  If you run this from a different port, you will need to update the CORS configuration in `simplerouter/api.py`.

### Setup for Open WebUI

[Open WebUI](https://openwebui.com/) is another option that people like using as a universal interface for AI.  It has not been tested with this app yet, but it should work with potentially some minor tweaks.  If you get this working please let Ola know, and we can create docs on how to set it up.
