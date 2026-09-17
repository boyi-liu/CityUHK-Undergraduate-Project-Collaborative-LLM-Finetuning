from alg.eventbase import EventBaseClient, EventBaseServer
from utils.time_utils import time_record


class Client(EventBaseClient):
    @time_record
    def run(self, model):
        # freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # then unfreeze lora_B only
        for name, param in model.named_parameters():
            if 'lora_B' in name:
                param.requires_grad = True

        self.trainer.train(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if 'lora_B' in k}


class Server(EventBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        # restrict global_lora to lora_B only, consistent with what clients upload
        self.global_lora = {k: v for k, v in self.global_lora.items() if 'lora_B' in k}
