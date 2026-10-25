from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        buckets = (0,.001,.002,.005,.01,.02,.05,.1,.2,.5,1,2,5)
        def hist(name, help):
            return Histogram('inferenceops_'+name, help, buckets=buckets, registry=self.registry)
        self.http = Counter('inferenceops_http_requests', 'Prediction HTTP terminal responses', ['status'], registry=self.registry)
        self.admitted = Counter('inferenceops_admitted', 'Accepted before upload/preprocessing', registry=self.registry)
        self.latency = hist('request_seconds', 'HTTP server latency, including failed responses')
        self.wait = hist('queue_wait_seconds', 'Prepared request enqueue to dispatch')
        self.accumulation = hist('accumulation_seconds', 'Intentional idle-worker accumulation; subset of queue wait')
        self.inference = hist('inference_seconds', 'Native execution duration per batch')
        self.batch = Histogram('inferenceops_batch_size', 'Executed batch sizes', buckets=(1,2,3,4,5,6,7,8), registry=self.registry)
        self.depth = Gauge('inferenceops_queue_depth', 'Prepared requests pending dispatch', registry=self.registry)
        self.rejections = Counter('inferenceops_rejections', 'Rejected work by boundary', ['stage'], registry=self.registry)
