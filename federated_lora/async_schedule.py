"""Reproducible asynchronous schedules, independent of training and torch."""
from __future__ import annotations

import copy
import heapq
import random


class AsyncSchedule:
    def __init__(self, config, initial, state=None):
        self.config = config
        self.settings = config['federation']['simulation']
        self.kind = self.settings['schedule']
        self.rng = random.Random(config['seed'])
        if state is not None:
            self.state = copy.deepcopy(state)
            self.rng.setstate(self.state['rng_state'])
            return
        self.state = {'now_us': 0, 'next_job_id': 0, 'queue': [],
                      'snapshots': {0: copy.deepcopy(initial)}, 'rng_state': self.rng.getstate()}
        if self.kind == 'arrivals':
            for cid in range(config['federation']['client_count']):
                self._dispatch(cid, initial, 0)
            self.state['snapshots'] = {}

    def _job(self, cid, adapter, version):
        job = {'job_id': self.state['next_job_id'], 'client': cid, 'downloaded_version': version,
               'dispatched_at_us': self.state['now_us'], 'adapter': copy.deepcopy(adapter)}
        self.state['next_job_id'] += 1
        return job

    def _dispatch(self, cid, adapter, version):
        job = self._job(cid, adapter, version)
        job['arrives_at_us'] = self.state['now_us'] + round(self.settings['client_job_seconds'][cid] * 1_000_000)
        heapq.heappush(self.state['queue'], (job['arrives_at_us'], job['job_id'], job))

    def next_job(self, version):
        if self.kind == 'arrivals':
            when, _, job = heapq.heappop(self.state['queue'])
            self.state['now_us'] = when
            return job
        # Section 5.2 samples a bounded stale version uniformly. Version zero
        # is available during startup; future or absent versions are never used.
        age = self.rng.randint(0, min(version, self.settings['max_staleness']))
        cid = self.rng.randrange(self.config['federation']['client_count'])
        job = self._job(cid, self.state['snapshots'][version - age], version - age)
        self.state['rng_state'] = self.rng.getstate()
        job['arrives_at_us'] = None
        return job

    def complete(self, job, adapter, version):
        if self.kind == 'arrivals':
            self._dispatch(job['client'], adapter, version)
        else:
            snapshots = self.state['snapshots']
            snapshots[version] = copy.deepcopy(adapter)
            minimum = max(0, version - self.settings['max_staleness'])
            self.state['snapshots'] = {v: a for v, a in snapshots.items() if v >= minimum}

    def checkpoint(self):
        return copy.deepcopy(self.state)
