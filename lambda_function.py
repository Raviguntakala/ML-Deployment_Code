import json
import boto3
import os
ENDPOINT_NAME = os.environ['EndPoint']
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    # TODO implement
    data = json.loads(json.dumps(event))
    payload = json.dumps(data)
    print("Invoking model endpoint")
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                   ContentType='application/json',
                                   Body=payload)
    result = json.loads(response['Body'].read().decode())    
    final_reponse = json.dumps(result, indent=2)
    print(final_reponse)
    return final_reponse