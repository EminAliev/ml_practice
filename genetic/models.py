from numpy import random
from random import randrange
from typing import List

from settings import Settings


class Member:

    def __init__(self, values, accuracy: float = 0.0):
        self._values = values
        self._accuracy = accuracy
        self._range_to_result = 0

    @property
    def values(self):
        return self._values

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value: float):
        self._accuracy = value

    @property
    def range_to_result(self) -> float:
        return self._range_to_result

    @range_to_result.setter
    def range_to_result(self, value: float):
        self._range_to_result = value


class MemberFactory:

    @staticmethod
    def create_new_generation_member(
            current_generation):

        probability_distribution = [current_generation[x].accuracy for x in range(len(current_generation))]
        parent1, parent2 = random.choice(current_generation, 2, p=probability_distribution)
        if parent1 == parent2:
            parent2 = random.choice(current_generation, 1, p=probability_distribution)[0]
        rnd = randrange(len(Settings.coefficients))

        dominant_parent = randrange(2)
        if dominant_parent == 0:
            new_generation_value = parent1.values
            new_generation_value[rnd] = parent2.values[rnd]
        else:
            new_generation_value = parent2.values
            new_generation_value[rnd] = parent1.values[rnd]

        if randrange(100) < 10:
            rand2 = randrange(len(Settings.coefficients))
            new_generation_value[rand2] = randrange(Settings.free_member)

        return Member(new_generation_value)


class Runner:

    @staticmethod
    def perform_generation_step(generation, inverse_coefficients_sum):
        for gen_index in range(Settings.generation_size):

            current_value = 0
            for c_index in range(len(Settings.coefficients)):
                current_value += generation[gen_index].values[c_index] * Settings.coefficients[c_index]

            abs_value = abs(current_value - Settings.free_member)

            if abs_value == 0:
                print("Answer: " + str(generation[gen_index].values))
                return True, inverse_coefficients_sum

            generation[gen_index].range_to_result = abs_value
            inverse_coefficients_sum += 1.0 / abs_value

        return False, inverse_coefficients_sum

    @staticmethod
    def recalculate_accuracy(
            generation,
            inverse_coefficients_sum):

        for s in range(len(generation)):
            generation[s].accuracy = (1.0 / generation[s].range_to_result) / inverse_coefficients_sum

    @staticmethod
    def create_generation():
        return \
            [
                Member([randrange(Settings.free_member) for _ in range(len(Settings.coefficients))])
                for _ in range(Settings.generation_size)
            ]
