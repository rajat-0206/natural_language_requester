import yaml
import json
import argparse
from collections import defaultdict

def simplify_schema(openapi_spec):
    """
    Simplifies an OpenAPI specification into a JSON format optimized for an AI model.

    Args:
        openapi_spec (dict): The parsed OpenAPI specification.

    Returns:
        list: A list of simplified API endpoint definitions.
    """
    simplified_api_list = []
    
    # In OpenAPI 3, reusable components are in the 'components' section.
    # We can pre-process these to handle '$ref' references more easily.
    components = openapi_spec.get('components', {})
    schemas = components.get('schemas', {})

    # Helper function to resolve schema references ($ref)
    def resolve_ref(ref):
        if not ref.startswith('#/components/schemas/'):
            # This script only handles local schema references.
            # You could extend this to handle file or URL references.
            print(f"Warning: Cannot resolve non-local schema reference: {ref}")
            return {}
        
        schema_name = ref.split('/')[-1]
        return schemas.get(schema_name, {})

    # Iterate over all paths defined in the OpenAPI spec
    for path, path_item in openapi_spec.get('paths', {}).items():
        # Iterate over all HTTP methods (get, post, put, etc.) for each path
        for method, operation in path_item.items():
            # Filter out non-operational keys like 'parameters' that can exist at the path level
            if method.lower() not in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options', 'trace']:
                continue

            # Use summary as a concise description, falling back to description
            description = operation.get('summary', operation.get('description', ''))
            
            api_definition = {
                "path": path,
                "method": method.upper(),
                "operationId": operation.get('operationId', 'N/A'),
                "description": description.strip(),
                "parameters": [],
                "requestBody": None
            }

            # 1. Process top-level and operation-level parameters (query, path, header, cookie)
            all_params = path_item.get('parameters', []) + operation.get('parameters', [])
            if all_params:
                for param in all_params:
                    # Resolve references if they exist
                    if '$ref' in param:
                        ref_path = param['$ref'].split('/')[1:] # e.g., #/components/parameters/userId -> ['components', 'parameters', 'userId']
                        param_def = components
                        for key in ref_path:
                            param_def = param_def.get(key, {})
                        param = param_def

                    param_schema = param.get('schema', {})
                    api_definition['parameters'].append({
                        "name": param.get('name'),
                        "in": param.get('in'),
                        "description": param.get('description', ''),
                        "required": param.get('required', False),
                        "type": param_schema.get('type', 'string')
                    })
            
            # 2. Process the request body
            request_body_spec = operation.get('requestBody', {})
            if request_body_spec:
                # Resolve reference if the whole requestBody is a reference
                if '$ref' in request_body_spec:
                    ref_path = request_body_spec['$ref'].split('/')[1:]
                    rb_spec = components
                    for key in ref_path:
                        rb_spec = rb_spec.get(key, {})
                    request_body_spec = rb_spec

                # Get the schema for 'application/json' content type
                json_content = request_body_spec.get('content', {}).get('application/json', {})
                if json_content:
                    body_schema = json_content.get('schema', {})
                    
                    # Resolve reference if the schema itself is a reference
                    if '$ref' in body_schema:
                        body_schema = resolve_ref(body_schema['$ref'])

                    payload = {}
                    required_fields = body_schema.get('required', [])
                    properties = body_schema.get('properties', {})
                    
                    for prop_name, prop_details in properties.items():
                        payload[prop_name] = {
                            "type": prop_details.get('type', 'any'),
                            "required": prop_name in required_fields,
                            "description": prop_details.get('description', '')
                        }
                    
                    if payload:
                        api_definition['requestBody'] = {
                            "required": request_body_spec.get('required', False),
                            "payload": payload
                        }

            simplified_api_list.append(api_definition)

    version = "1.0.0"
    schema = {
        "version": version,
        "apis": simplified_api_list
    }

    return schema

def main():
    """Main function to run the script."""

    print(f"Loading OpenAPI spec from: goldcast_external_api.yaml")
    try:
        with open('goldcast_external_api.yaml', 'r', encoding='utf-8') as f:
            # Use yaml.safe_load() for security
            openapi_spec = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at goldcast_external_api.yaml")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    print("Simplifying schema...")
    schema = simplify_schema(openapi_spec)

    print(f"Saving simplified schema to: schema.json")
    try:
        with open('schema.json', 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2)
    except IOError as e:
        print(f"Error writing to output file: {e}")
        return
        
    print("\nConversion complete!")
    print(f"Processed {len(schema['apis'])} API endpoints.")

if __name__ == '__main__':
    main()
