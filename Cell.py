from dataclasses import dataclass


@dataclass
class Cell:
    image: dict
    contains_digit: bool
    digit: int
