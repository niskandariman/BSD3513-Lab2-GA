"""
UNIVERSITI MALAYSIA PAHANG AL-SULTAN ABDULLAH (UMPSA)
Course   : BSD3513 / BSD2513 Artificial Intelligence
Lab      : Lab Report 2 - Genetic Algorithm Bit Pattern Search
Author   : Nor Iskandar Iman bin Noorazlind
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if Streamlit is running the script
IS_STREAMLIT = "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv)
if IS_STREAMLIT:
    import streamlit as st


# ==========================================
# 1. CORE GENETIC ALGORITHM ENGINE
# ==========================================

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int = 300,
        chromosome_len: int = 80,
        target_ones: int = 50,
        max_fitness: int = 80,
        generations: int = 50,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.0125,
        elite_count: int = 2,
        seed: int = None,
    ):
        self.pop_size = pop_size
        self.chromosome_len = chromosome_len
        self.target_ones = target_ones
        self.max_fitness = max_fitness
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count

        if seed is not None:
            np.random.seed(seed)

    def initialize_population(self) -> np.ndarray:
        """Generates initial population of shape (pop_size, chromosome_len)."""
        return np.random.randint(0, 2, size=(self.pop_size, self.chromosome_len), dtype=np.int8)

    def fitness_function(self, chromosome: np.ndarray) -> int:
        """
        Fitness formula:
        Reaches maximum value (80) when number of ones equals 50.
        Fitness = 80 - |count_ones - 50|
        """
        num_ones = int(np.sum(chromosome))
        return self.max_fitness - abs(num_ones - self.target_ones)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Computes fitness for every individual in the population."""
        ones_count = np.sum(population, axis=1)
        return self.max_fitness - np.abs(ones_count - self.target_ones)

    def tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
        """Tournament selection of size k."""
        selected_idx = np.random.randint(0, self.pop_size, size=k)
        best_idx = selected_idx[np.argmax(fitnesses[selected_idx])]
        return population[best_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Two-point crossover."""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        cx_points = np.sort(np.random.choice(self.chromosome_len, size=2, replace=False))
        cp1, cp2 = cx_points[0], cx_points[1]

        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        offspring1[cp1:cp2] = parent2[cp1:cp2]
        offspring2[cp1:cp2] = parent1[cp1:cp2]

        return offspring1, offspring2

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Bit-flip mutation based on gene mutation probability."""
        mutation_mask = np.random.rand(self.chromosome_len) < self.mutation_rate
        chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
        return chromosome

    def run(self):
        """Executes the genetic algorithm across generations."""
        population = self.initialize_population()
        history = {
            "generation": [],
            "best_fitness": [],
            "avg_fitness": [],
            "best_ones_count": [],
            "best_chromosome": []
        }

        best_overall_individual = None
        best_overall_fitness = -1

        for gen in range(self.generations):
            fitnesses = self.evaluate_population(population)

            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_ind = population[gen_best_idx].copy()
            gen_avg_fitness = np.mean(fitnesses)
            gen_best_ones = np.sum(gen_best_ind)

            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_individual = gen_best_ind

            history["generation"].append(gen + 1)
            history["best_fitness"].append(gen_best_fitness)
            history["avg_fitness"].append(gen_avg_fitness)
            history["best_ones_count"].append(gen_best_ones)
            history["best_chromosome"].append("".join(map(str, gen_best_ind)))

            # Elitism: retain top individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            next_population = [population[idx].copy() for idx in sorted_indices[:self.elite_count]]

            # Reproduction
            while len(next_population) < self.pop_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                next_population.append(c1)
                if len(next_population) < self.pop_size:
                    next_population.append(c2)

            population = np.array(next_population)

        final_fitnesses = self.evaluate_population(population)
        final_best_idx = np.argmax(final_fitnesses)
        if final_fitnesses[final_best_idx] > best_overall_fitness:
            best_overall_fitness = final_fitnesses[final_best_idx]
            best_overall_individual = population[final_best_idx]

        return history, best_overall_individual, best_overall_fitness


# ==========================================
# 2. STREAMLIT WEB APPLICATION INTERFACE
# ==========================================

def render_streamlit_app():
    st.set_page_config(page_title="AI Lab 2 - Genetic Algorithm", layout="wide", page_icon="🧬")
    
    st.title("🧬 Genetic Algorithm Bit Pattern Search")
    st.markdown("""
    **Course:** BSD3513 Artificial Intelligence | **Topic:** Chapter 2 Search Algorithms  
    **Task:** Evolve an 80-bit individual reaching optimal fitness ($80$) when the count of ones equals exactly $50$.
    """)
    st.divider()

    # Sidebar Parameters
    st.sidebar.header("⚙️ GA Hyperparameters")
    pop_size = st.sidebar.number_input("Population Size", min_value=50, max_value=1000, value=300, step=50)
    generations = st.sidebar.number_input("Generations", min_value=10, max_value=200, value=50, step=5)
    chromosome_len = st.sidebar.number_input("Chromosome Length", min_value=20, max_value=200, value=80, step=10)
    target_ones = st.sidebar.number_input("Target Ones Count", min_value=1, max_value=chromosome_len, value=50, step=1)
    max_return_val = st.sidebar.number_input("Target Max Value", min_value=10, max_value=200, value=80, step=5)
    
    crossover_rate = st.sidebar.slider("Crossover Probability (Pc)", 0.0, 1.0, 0.85, 0.05)
    mutation_rate = st.sidebar.slider("Mutation Probability (Pm)", 0.001, 0.1, 0.0125, 0.0025, format="%.4f")
    elite_count = st.sidebar.slider("Elitism Count", 0, 10, 2, 1)

    if st.sidebar.button("🚀 Run Genetic Algorithm", type="primary"):
        ga = GeneticAlgorithm(
            pop_size=pop_size,
            chromosome_len=chromosome_len,
            target_ones=target_ones,
            max_fitness=max_return_val,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_count=elite_count,
        )

        with st.spinner("Simulating evolutionary cycles..."):
            history, best_ind, best_fit = ga.run()

        df_history = pd.DataFrame(history)

        # Top metric cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Optimal Fitness Achieved", f"{best_fit} / {max_return_val}")
        col2.metric("Target Ones", f"{target_ones}")
        col3.metric("Best Ones Count", f"{int(np.sum(best_ind))}")
        col4.metric("Total Generations", f"{generations}")

        st.subheader("📈 Convergence Diagnostics")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df_history["generation"], df_history["best_fitness"], label="Best Fitness", color="#00C49F", linewidth=2.5)
        ax.plot(df_history["generation"], df_history["avg_fitness"], label="Average Fitness", color="#FFBB28", linestyle="--")
        ax.axhline(y=max_return_val, color="red", linestyle=":", label="Theoretical Maximum (80)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Score")
        ax.set_title("Fitness Progression across Evolutionary Generations")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

        st.subheader("🧬 Best Individual Bitstream Visualization")
        bit_str = "".join(map(str, best_ind))
        st.code(bit_str, language="text")

        # Visual representation of chromosome as an 8x10 matrix
        fig_mat, ax_mat = plt.subplots(figsize=(10, 2))
        matrix = best_ind.reshape(8, 10)
        cax = ax_mat.imshow(matrix, cmap="Blues", aspect="auto")
        for i in range(8):
            for j in range(10):
                ax_mat.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black" if matrix[i, j] == 0 else "white", fontsize=8)
        ax_mat.set_title("Chromosome Structure (8x10 Grid representation)")
        ax_mat.set_xticks([])
        ax_mat.set_yticks([])
        st.pyplot(fig_mat)

        with st.expander("📋 Detailed Generation History Table"):
            st.dataframe(df_history[["generation", "best_fitness", "avg_fitness", "best_ones_count"]], use_container_width=True)


# ==========================================
# 3. CLI STANDALONE EXECUTION
# ==========================================

def run_cli():
    print("=" * 70)
    print("UMPSA BSD3513 - Search Algorithms: Genetic Algorithm (CLI Mode)")
    print("=" * 70)

    ga = GeneticAlgorithm(
        pop_size=300,
        chromosome_len=80,
        target_ones=50,
        max_fitness=80,
        generations=50,
        crossover_rate=0.85,
        mutation_rate=0.0125,
        elite_count=2,
    )

    history, best_chromosome, best_fitness = ga.run()

    print(f"\nOptimization Completed!")
    print(f"Generations Run      : {ga.generations}")
    print(f"Population Size      : {ga.pop_size}")
    print(f"Chromosome Length    : {ga.chromosome_len}")
    print(f"Best Fitness Achieved: {best_fitness} / {ga.max_fitness}")
    print(f"Count of Ones in Best: {np.sum(best_chromosome)} (Target: {ga.target_ones})")
    print(f"\nOptimal Bit Pattern:\n{''.join(map(str, best_chromosome))}")
    print("=" * 70)


if __name__ == "__main__":
    if IS_STREAMLIT:
        render_streamlit_app()
    else:
        run_cli()
        