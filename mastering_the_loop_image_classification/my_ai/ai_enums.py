from enum import IntEnum


class Scheduler(IntEnum):
    NoScheduling = 0
    StepLR = 1
    CosineAnnealing = 2