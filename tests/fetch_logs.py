import subprocess
import json

q = 'resource.type="cloud_run_revision" AND resource.labels.service_name="ramp-bot" AND timestamp>="2026-05-02T14:30:00Z" AND timestamp<="2026-05-03T00:00:00Z"'
try:
    out = subprocess.check_output(['gcloud.cmd', 'logging', 'read', q, '--limit', '200', '--format', 'json'])
    logs = json.loads(out)
    for x in logs:
        if 'textPayload' in x:
            print(f"[{x['timestamp']}] {x['textPayload']}")
except Exception as e:
    print("Error:", e)
