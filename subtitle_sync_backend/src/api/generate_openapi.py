import json
import os

from src.api.main import app

"""
Utility script to generate the OpenAPI schema into interfaces/openapi.json.
Run this after modifying routes or models to refresh the spec.
"""

# Get the OpenAPI schema
openapi_schema = app.openapi()

# Write to file
output_dir = "interfaces"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "openapi.json")

with open(output_path, "w") as f:
    json.dump(openapi_schema, f, indent=2)
