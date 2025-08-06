import app.generator

def lambda_handler(event, context):
    query = event.get("queryStringParameters", {}).get("query", "")
    k = int(event.get("queryStringParameters", {}).get("k", 3))

    response = app.generator.generate_response(query, k)

    return {
        "statusCode": 200,
        "body": response,
    }
