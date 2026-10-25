"""Event-loop-owned bounded FIFO, one consumer and one native call in flight."""
import asyncio
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from fastapi import HTTPException


@dataclass
class Entry:
    tensor: np.ndarray
    future: asyncio.Future
    queued: float
    deadline: float
    cancelled: asyncio.Event
    accumulation: float = 0.0


class Batcher:
    def __init__(self, model, executor, mode='immediate', size=8, delay=.01, capacity=8, metrics=None):
        if mode not in {'immediate', 'fixed', 'dynamic'} or not 1 <= size <= 8 or delay <= 0 or capacity < 1:
            raise ValueError('Invalid scheduler configuration')
        self.metrics = metrics
        self.model, self.executor, self.mode = model, executor, mode
        self.size, self.delay, self.capacity = size, delay, capacity
        self.queue = []
        self.changed = asyncio.Event()
        self.closed = False
        self.origin = perf_counter()
        self.worker = asyncio.create_task(self.run())

    async def submit(self, tensor, deadline=float('inf'), cancelled=None):
        if self.closed or len(self.queue) >= self.capacity:
            if self.metrics:
                self.metrics.rejections.labels('queue').inc()
            raise HTTPException(503, 'Inference queue unavailable')
        future = asyncio.get_running_loop().create_future()
        self.queue.append(Entry(tensor, future, perf_counter(), deadline, cancelled or asyncio.Event()))
        if self.metrics:
            self.metrics.depth.set(len(self.queue))
        self.changed.set()
        return await future

    def prune(self):
        retained = []
        for entry in self.queue:
            if entry.cancelled.is_set() or perf_counter() >= entry.deadline:
                if not entry.future.done():
                    entry.future.set_exception(HTTPException(504, 'Request expired or cancelled'))
            else:
                retained.append(entry)
        self.queue = retained
        if self.metrics:
            self.metrics.depth.set(len(self.queue))

    async def run(self):
        fixed_ready = False
        while True:
            self.prune()
            if not self.queue:
                if self.closed:
                    return
                self.changed.clear()
                await self.changed.wait()
                fixed_ready = False
                continue
            now = perf_counter()
            delay = 0
            if self.mode == 'fixed' and not fixed_ready:
                delay = self.delay - (now-self.origin) % self.delay
            elif self.mode == 'dynamic' and len(self.queue) < self.size:
                delay = max(0, self.queue[0].queued+self.delay-now)
            target = now + delay
            if delay > 0:
                delay = min(delay, max(0, min(e.deadline for e in self.queue)-now))
                self.changed.clear()
                waiting = list(self.queue)
                wait_started = perf_counter()
                try:
                    await asyncio.wait_for(self.changed.wait(), delay)
                    continue
                except TimeoutError:
                    fixed_ready = self.mode == 'fixed' and perf_counter() >= target
                finally:
                    elapsed_wait = perf_counter()-wait_started
                    for entry in waiting:
                        entry.accumulation += elapsed_wait
                self.prune()
                if not self.queue:
                    continue
                if self.mode != 'fixed' or not fixed_ready:
                    continue
            entries = self.queue[:1 if self.mode == 'immediate' else self.size]
            del self.queue[:len(entries)]
            fixed_ready = False
            if self.metrics:
                self.metrics.depth.set(len(self.queue))
            entered = perf_counter()
            try:
                loop = asyncio.get_running_loop()
                if self.mode == 'immediate':
                    result, elapsed = await loop.run_in_executor(self.executor, self.model.predict, entries[0].tensor)
                    results = [result]
                else:
                    logits, elapsed = await loop.run_in_executor(self.executor, self.model.run, np.stack([e.tensor for e in entries]))
                    results = []
                    for row in logits:
                        probabilities = np.exp(row-row.max())
                        probabilities /= probabilities.sum()
                        results.append([{'class_id': int(i), 'label': self.model.categories[i], 'probability': float(probabilities[i])}
                                        for i in np.argsort(-probabilities, kind='stable')[:5]])
                if self.metrics:
                    self.metrics.inference.observe(elapsed / 1000)
                    self.metrics.batch.observe(len(entries))
                    for entry in entries:
                        self.metrics.wait.observe(entered-entry.queued)
                        self.metrics.accumulation.observe(entry.accumulation)
                for entry, result in zip(entries, results):
                    if not entry.future.done():
                        if entry.cancelled.is_set() or perf_counter() >= entry.deadline:
                            entry.future.set_exception(HTTPException(504, 'Request expired or cancelled'))
                        else:
                            entry.future.set_result((result, elapsed, (entered-entry.queued)*1000, entry.accumulation*1000, len(entries)))
            except Exception as exc:
                for entry in entries:
                    if not entry.future.done():
                        entry.future.set_exception(exc)

    async def close(self):
        self.closed = True
        for entry in self.queue:
            if not entry.future.done():
                entry.future.set_exception(HTTPException(503, 'Server shutting down'))
        self.queue.clear()
        if self.metrics:
            self.metrics.depth.set(0)
        self.changed.set()
        await self.worker
