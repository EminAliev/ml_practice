from models import MemberFactory, Runner
from settings import Settings


def main():
    print(f'Coefficients for unknowns: {Settings.coefficients}')
    print(f'Free member: {Settings.free_member}')

    inverse_coefficients_sum = 0.0
    generation = Runner.create_generation()

    solutions_is_found, inverse_coefficients_sum = Runner.perform_generation_step(
        generation,
        inverse_coefficients_sum)

    Runner.recalculate_accuracy(generation, inverse_coefficients_sum)

    current_iteration = 0
    while not solutions_is_found and current_iteration < Settings.max_iterations:
        inverse_coefficients_sum = 0.0
        generation = \
            [
                MemberFactory.create_new_generation_member(generation)
                for _ in range(Settings.generation_size)
            ]

        solutions_is_found, inverse_coefficients_sum = Runner.perform_generation_step(
            generation,
            inverse_coefficients_sum)

        if solutions_is_found:
            break

        Runner.recalculate_accuracy(generation, inverse_coefficients_sum)
        current_iteration += 1

    if solutions_is_found:
        print(f'Current iteration: {current_iteration}')
    else:
        print(f'Maximum number of iterations exceeded ({Settings.max_iterations})')


if __name__ == "__main__":
    main()
