import yaml
import os
import json
import argparse
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import sys
import io

# Load configuration from config.yml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Set environment variables
os.environ['MODEL_PATH'] = config['model']['path']
os.environ['MODEL_VERSION'] = config['model']['version']
os.environ['BATCH_SIZE'] = str(config['inference']['batch_size'])
os.environ['DEVICE'] = config['inference']['device']
os.environ['API_KEY'] = config['api']['key']
os.environ['API_ENDPOINT'] = config['api']['endpoint']

# Access configuration parameters from environment variables
model_path = os.getenv('MODEL_PATH')
model_version = os.getenv('MODEL_VERSION')
batch_size = int(os.getenv('BATCH_SIZE'))
device = os.getenv('DEVICE')
api_key = os.getenv('API_KEY')
api_endpoint = os.getenv('API_ENDPOINT')

# Ensure API key is provided
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# Initialize the client
client = ChatCompletionsClient(
    endpoint=api_endpoint,
    credential=AzureKeyCredential(api_key)
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run inference with Azure AI.")
parser.add_argument('-p', '--payload', required=True, help="Path to the payload JSON file.")
parser.add_argument('-o', '--output', required=True, help="Path to the output file.")
args = parser.parse_args()

# Redirect print statements to the output file with UTF-8 encoding
output_file = args.output
sys.stdout = io.TextIOWrapper(open(output_file, 'wb'), encoding='utf-8')

# Read and parse the payload file
payload_file = args.payload
if not os.path.exists(payload_file):
    raise FileNotFoundError(f"The specified payload file '{payload_file}' does not exist.")

try:
    with open(payload_file, 'r', encoding='utf-8') as file:
        payload = json.load(file)
except json.JSONDecodeError as e:
    raise Exception(f"Failed to parse JSON from file '{payload_file}': {e}")

# Run inference
response = client.complete(payload)

# Print the response
print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Usage:")
print("    Prompt tokens:", response.usage.prompt_tokens)
print("    Total tokens:", response.usage.total_tokens)
print("    Completion tokens:", response.usage.completion_tokens)

# Close the output file
sys.stdout.close()
