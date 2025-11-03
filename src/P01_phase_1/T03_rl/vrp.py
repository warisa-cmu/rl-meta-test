import numpy as np


class VRP:
    def __init__(
        self,
        population_size,
        dimensions,
        bounds,
        Mutation_rate,
        Crossover_rate,
        distance,
        max_iteration=1000,
        step_iteration=100,
    ):
        # Input params
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.Mutation_rate = Mutation_rate
        self.Crossover_rate = Crossover_rate
        self.max_iteration = max_iteration
        self.distance = distance
        self.step_iteration = step_iteration

        # Internal params
        self.global_solution = np.array([])
        self.F = Mutation_rate[0]
        self.CR = Crossover_rate[0]
        self.current_cost = np.array([])
        self.kwargs = {"distance": self.distance}
        self.current_iteration = 0
        self.delta_F = 0.01

        # Derived internal
        self.population = None
        self.Upperbound_Mutation = None
        self.Lowerbound_Mutation = None
        self.Upperbound_Crossover_rate = None
        self.Lowerbound_Crossover_rate = None

    def reset(self):
        # Initialize population
        self.population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.population_size, len(self.bounds)),
        )
        self.Upperbound_Mutation = self.Mutation_rate[1]
        self.Lowerbound_Mutation = self.Mutation_rate[0]
        self.Upperbound_Crossover_rate = self.Crossover_rate[1]
        self.Lowerbound_Crossover_rate = self.Crossover_rate[0]
        self.F = self.Mutation_rate[0]
        self.CR = self.Crossover_rate[0]
        self.current_iteration = 0

    def preserving_strategy(self, X, **kwargs):
        # distance matrix
        distance = kwargs["distance"]
        # total distance starts from zero km.
        total_distance = 0
        # Vehicle travel from depot to customer i
        total_distance += distance[0][X[0]]
        # Total distance of routing solution
        for i in range(len(X) - 1):
            total_distance += distance[X[i]][X[i + 1]]
        # Vehicle returns to depot
        total_distance += distance[X[-1]][0]
        # Return total distance (km.) that vehicle traveled
        return total_distance

    def f_per_particle(self, m, **kwargs):
        X = m  # Sequence
        obj_val = self.preserving_strategy(X, **kwargs)  # Call Preserving strategy.
        return obj_val

    def objective_func(self, x, **kwargs):
        """Decoding of each particles for obtaining routing solutions by argsort()"""
        seq = x.argsort() + 1
        """Calculate objective function for obtaining objective value of each particle"""
        j = self.f_per_particle(seq, **kwargs)
        return np.array(j)

    def evolve(self, n_iteration):
        self.current_iteration = self.current_iteration + n_iteration

        Upperbound_Mutation = self.Upperbound_Mutation
        Lowerbound_Mutation = self.Lowerbound_Mutation
        Upperbound_Crossover_rate = self.Upperbound_Crossover_rate
        Lowerbound_Crossover_rate = self.Lowerbound_Crossover_rate
        population_size = self.population_size
        population = self.population
        bounds = self.bounds
        max_generations = n_iteration

        for _ in range(max_generations):
            # print(f'Iteration {generation}')
            current_cost = np.array([])
            self.delta_F = (Upperbound_Mutation - Lowerbound_Mutation) / max_generations

            # self.F += (Upperbound_Mutation - Lowerbound_Mutation) / max_generations
            self.CR += (
                Upperbound_Crossover_rate - Lowerbound_Crossover_rate
            ) / max_generations
            for i in range(population_size):
                # Mutation
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = population[i] + self.F * (b - c)

                # Crossover
                crossover_prob = np.random.rand(len(bounds))
                trial = np.where(crossover_prob < self.CR, mutant, population[i])

                # Selection
                fitness_trial = self.objective_func(trial, **self.kwargs)
                fitness_current = self.objective_func(population[i], **self.kwargs)

                if fitness_trial < fitness_current:
                    population[i] = trial
                    current_cost = np.insert(
                        current_cost, len(current_cost), fitness_trial
                    )
                else:
                    current_cost = np.insert(
                        current_cost, len(current_cost), fitness_current
                    )
                # print(f"population {i}")
                # print(f"current_cost{current_cost}")
                # print("---------" * 30)
            best_index_plot = current_cost[np.argmin(current_cost)]
            self.global_solution = np.insert(
                self.global_solution, len(self.global_solution), best_index_plot
            )
        # Find the best solution
        best_index = np.argmin(
            [
                self.objective_func(individual, **self.kwargs)
                for individual in population
            ]
        )
        best_solution = population[best_index]

        return best_solution, self.global_solution

    def get_best_solution(self):
        return np.random.random()

    def is_exceed_max_iteration(self):
        return self.current_iteration > self.max_iteration

    def change_F(self, mode):
        if mode == "INCREASE":
            self.F += self.delta_F
        elif mode == "DECREASE":
            self.F -= self.delta_F
            if self.F < 0:
                self.F += self.delta_F  # Return to previous value
        else:
            raise Exception("Invalid Option")
