from datetime import timedelta

from alg.eventbase import EventBaseClient, EventBaseServer, ClientState, Event, EventType
from utils.time_utils import time_record


class AsyncEventBaseClient(EventBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.start_round = 0

    @time_record
    def run(self, model):
        super().run(model)


class AsyncEventBaseServer(EventBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.decay = args.decay
        self._active_clients = set()
        self._cur_client = None

    # ── override run to skip pre-sampling ────────────────────────────────────

    def run(self) -> None:
        """Process events until one async aggregation completes."""
        self._round_complete = False
        self._uploads_this_round = []
        self.round += 1
        while self.event_queue and not self._round_complete:
            event = self.event_queue.pop()
            print(f'[Pop Event] Type, {event.type}; Event time, {event.time}; Event client, {event.client.id}')
            self.wall_clock_time = event.time
            if self.wall_clock_time > self._deadline:
                print(f'[{self.wall_clock_time}] Session deadline reached after round {self.round - 1}, stopping.')
                self.early_break = True
                break
            self._dispatch(event)

    # ── event handlers ────────────────────────────────────────────────────────

    def _try_assign(self, client: AsyncEventBaseClient) -> None:
        """Assign training if capacity allows (up to sample_num concurrent)."""
        if client.state != ClientState.AVAILABLE:
            return
        if len(self._active_clients) >= self.sample_num:
            return

        print(f'[{self.wall_clock_time}] Assigning training to client {client.id}.')
        client.start_round = self.round
        client.state = ClientState.TRAINING
        self._active_clients.add(client)
        client._train_start_time = self.wall_clock_time
        client._train_window_end = self._get_client_window_close(client)
        client.run(self.model)
        self.model.load_state_dict(self.global_lora, strict=False)
        print(f'[{self.wall_clock_time}] Training cost {client.training_time:.2f} seconds.')
        finish_at = self.wall_clock_time + timedelta(seconds=client.training_time)
        
        if not self._window_closes_before_upload(client, finish_at):
            self.event_queue.push(Event(finish_at, EventType.TRAINING_DONE, client))
    

    def _on_window_close(self, event: Event) -> None:
        self._active_clients.discard(event.client)
        super()._on_window_close(event)

    def _on_upload_done(self, event: Event) -> None:
        client: AsyncEventBaseClient = event.client
        self._active_clients.discard(client)
        # One upload per window: keep the client out of the AVAILABLE pool so the
        # refill loop below won't reassign it until its window reopens.
        client.state = ClientState.DONE
        print(f'[{self.wall_clock_time}] Client {client.id}: upload complete.')

        # Async: aggregate immediately with this single client
        self._cur_client = client
        self.aggregate()
        self._round_complete = True

        # Refill capacity with any available clients
        for c in self.clients:
            if c.state == ClientState.AVAILABLE:
                self._try_assign(c)

    # ── aggregation ───────────────────────────────────────────────────────────

    def aggregate(self) -> None:
        alpha = self.decay * self.weight_decay()
        from collections import defaultdict
        server_lora = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
        client_lora = self._cur_client.lora
        aggregated = defaultdict(lambda: 0)
        for k in server_lora:
            aggregated[k] = alpha * client_lora[k] + (1 - alpha) * server_lora[k]
        self.global_lora = aggregated
        self.model.load_state_dict(self.global_lora, strict=False)
        print(f'[{self.wall_clock_time}] Aggregated model updated (round {self.round}).')

    def weight_decay(self) -> float:
        return 1
