import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# ----------------------------
# 1. Definisi Sistem Kontrol Oven dan PID Controller
# ----------------------------

# Parameter Sistem Oven
T_env = 25.0      # Suhu lingkungan dalam derajat Celsius
tau = 100.0       # Konstanta waktu sistem
K = 0.5           # Gain sistem

# Setpoint suhu oven
setpoint = 200.0  # Derajat Celsius

# Waktu simulasi
time_steps = 500  # Jumlah langkah waktu
dt = 1.0          # Interval waktu (detik)

def simulate_pid(P_gain, I_gain, D_gain):
    """
    Simulasi sistem kontrol PID untuk mengatur suhu oven.

    Args:
        P_gain (float): Parameter Proportional
        I_gain (float): Parameter Integral
        D_gain (float): Parameter Derivative

    Returns:
        float: Total Squared Error (TSE) selama simulasi
    """
    T = T_env  # Inisialisasi suhu awal oven
    integral = 0.0
    previous_error = 0.0
    TSE = 0.0  # Total Squared Error

    for _ in range(time_steps):
        error = setpoint - T
        integral += error * dt
        derivative = (error - previous_error) / dt
        P = P_gain * error + I_gain * integral + D_gain * derivative
        P = max(0.0, min(1.0, P))  # Membatasi P antara 0 dan 1

        # Update suhu oven berdasarkan model
        dTdt = -(T - T_env) / tau + K * P
        T += dTdt * dt

        # Akumulasi TSE
        TSE += error**2

        previous_error = error

    return TSE

# ----------------------------
# 2. Definisi Fungsi Objektif untuk Optimasi
# ----------------------------

def objective(individual):
    """
    Fungsi objektif untuk optimasi DE. Menghitung Total Squared Error (TSE).

    Args:
        individual (list): [P, I, D] parameter PID

    Returns:
        tuple: (TSE,)
    """
    P, I, D = individual
    TSE = simulate_pid(P, I, D)
    return (TSE,)

# ----------------------------
# 3. Implementasi Algoritma Differential Evolution dengan DEAP
# ----------------------------

# Definisikan Fitness dan Individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimalkan TSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Atribut Individu: Nilai P, I, D dalam rentang tertentu
toolbox.register("attr_P", random.uniform, 0.0, 10.0)
toolbox.register("attr_I", random.uniform, 0.0, 10.0)
toolbox.register("attr_D", random.uniform, 0.0, 10.0)

# Struktur Individu
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_P, toolbox.attr_I, toolbox.attr_D), n=1)

# Populasi
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operator Evaluasi
toolbox.register("evaluate", objective)

# Operator Mutasi dan Crossover
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------------------------
# 4. Parameter Algoritma DE
# ----------------------------

population_size = 50
num_generations = 100
crossover_prob = 0.7
mutation_prob = 0.2

# ----------------------------
# 5. Simulasi dan Visualisasi
# ----------------------------

def main():
    random.seed(42)  # Untuk reprodusibilitas

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)  # Menyimpan individu terbaik

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Jalankan Algoritma Evolusi
    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=crossover_prob,
                                   mutpb=mutation_prob,
                                   ngen=num_generations,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    best_ind = hof[0]
    print(f"\nParameter PID terbaik: P={best_ind[0]:.4f}, I={best_ind[1]:.4f}, D={best_ind[2]:.4f}")
    print(f"Total Squared Error terbaik: {best_ind.fitness.values[0]:.4f}")

    # Visualisasi Evolusi Fitness
    gen = log.select("gen")
    min_fitness = log.select("min")
    avg_fitness = log.select("avg")

    plt.figure(figsize=(10, 5))
    plt.plot(gen, min_fitness, label="Min Fitness")
    plt.plot(gen, avg_fitness, label="Average Fitness")
    plt.xlabel("Generasi")
    plt.ylabel("Fitness (TSE)")
    plt.title("Evolusi Fitness Selama Optimasi")
    plt.legend()
    plt.grid()
    plt.show()

    # Simulasi dengan Parameter Terbaik
    simulate_and_plot(best_ind[0], best_ind[1], best_ind[2])

def simulate_and_plot(P, I, D):
    """
    Simulasi sistem kontrol PID dengan parameter terbaik dan plot hasilnya.

    Args:
        P (float): Parameter Proportional
        I (float): Parameter Integral
        D (float): Parameter Derivative
    """
    T = T_env
    integral = 0.0
    previous_error = 0.0
    temperatures = []
    setpoints = []

    for _ in range(time_steps):
        error = setpoint - T
        integral += error * dt
        derivative = (error - previous_error) / dt
        P_control = P * error + I * integral + D * derivative
        P_control = max(0.0, min(1.0, P_control))  # Batasan P antara 0 dan 1

        # Update suhu oven
        dTdt = -(T - T_env) / tau + K * P_control
        T += dTdt * dt

        temperatures.append(T)
        setpoints.append(setpoint)

        previous_error = error

    # Plot Hasil Simulasi
    plt.figure(figsize=(12, 6))
    plt.plot(range(time_steps), temperatures, label="Suhu Oven")
    plt.plot(range(time_steps), setpoints, 'r--', label="Setpoint")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Suhu (°C)")
    plt.title("Simulasi Sistem Kontrol PID dengan Parameter Terbaik")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Error
    errors = [setpoint - temp for temp in temperatures]
    plt.figure(figsize=(12, 6))
    plt.plot(range(time_steps), errors, label="Error (Setpoint - Suhu)")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Error (°C)")
    plt.title("Error Seiring Waktu dengan Parameter PID Terbaik")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
