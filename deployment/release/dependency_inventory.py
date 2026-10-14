"""Print installed distribution licensing metadata without importing packages.

Run separately with model and MCP interpreters. Metadata is evidence for review,
not a substitute for bundled LICENSE/NOTICE files or permission to redistribute.
"""
import importlib.metadata as metadata
import json

rows = []
for dist in metadata.distributions():
    m = dist.metadata
    license_files = []
    for f in dist.files or ():
        if any(word in str(f).lower() for word in ('license', 'notice', 'copying')):
            license_files.append(str(f))
    rows.append({'name': m['Name'], 'version': dist.version,
                 'license_expression': m.get('License-Expression'),
                 'license_metadata': m.get('License'),
                 'license_classifiers': [c for c in m.get_all('Classifier', []) if c.startswith('License')],
                 'project_urls': m.get_all('Project-URL', []), 'license_files': license_files})
print(json.dumps(sorted(rows, key=lambda r: r['name'].lower()), ensure_ascii=False, indent=2))
