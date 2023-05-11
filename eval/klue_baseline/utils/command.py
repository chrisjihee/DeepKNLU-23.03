from typing import List


class Command:

    Train = "train"
    Valid = "valid"
    Test = "test"

    @staticmethod
    def tolist() -> List[str]:
        return [Command.Train, Command.Valid, Command.Test]
