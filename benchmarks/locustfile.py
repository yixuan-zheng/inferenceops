"""Closed-user workload; spawn rate is not an offered request rate."""
from pathlib import Path
from locust import HttpUser, task, constant
PAYLOAD = (Path(__file__).resolve().parents[1]/'benchmarks/fixtures/input.png').read_bytes()

class InferenceUser(HttpUser):
    wait_time = constant(0)
    @task
    def predict(self):
        with self.client.post('/predict', data=PAYLOAD, headers={'Content-Type':'image/png'}, catch_response=True) as response:
            if response.status_code == 200 and len(response.json().get('predictions',[])) != 5:
                response.failure('Invalid prediction payload')

# Locust's periodic CSV snapshot can lag its final console totals.
# Save exact final counters at shutdown instead of treating a stale snapshot as final.
from locust import events
import json

@events.quitting.add_listener
def final_counts(environment, **kwargs):
    total = environment.stats.total
    output = Path(__file__).resolve().parents[1]/'results/generated/locust-final.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'requests': total.num_requests, 'failures':total.num_failures,
        'successes': total.num_requests-total.num_failures,
        'errors': [e.to_dict() for e in environment.stats.errors.values()],
        'note':'Final closed-user totals, distinct from periodic CSV snapshots. Failure-inclusive percentiles are not successful-request latency.'},indent=2)+'\n')
