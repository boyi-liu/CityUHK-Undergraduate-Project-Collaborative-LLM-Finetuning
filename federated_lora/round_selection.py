"""Select the earliest highest score and stop at the first strict decrease."""
from dataclasses import dataclass, asdict


@dataclass
class AccuracySelection:
    count: int
    previous_correct: int
    best_correct: int
    best_round: int = 0
    last_round: int = 0
    stopped: bool = False

    @classmethod
    def initial(cls, correct, count):
        if type(count) is not int or type(correct) is not int or not 0 <= correct <= count or count < 1:
            raise ValueError("Invalid initial accuracy counts")
        return cls(count, correct, correct)

    def observe(self, round_number, correct, count):
        if self.stopped or round_number != self.last_round + 1:
            raise ValueError("Round selection requires consecutive rounds before stopping")
        if count != self.count or type(correct) is not int or not 0 <= correct <= count:
            raise ValueError("Every round must score the same questions")
        decreased = correct < self.previous_correct
        if correct > self.best_correct:
            self.best_correct, self.best_round = correct, round_number
        self.previous_correct, self.last_round = correct, round_number
        self.stopped = decreased
        return decreased

    def to_dict(self):
        return asdict(self)
