

import time
import random
import copy
import math
import robotica
from modelo_final import DEFAULT_CONFIG

# Se necesita acceso a las variables globales del módulo para resetearlas
import modelo_final as m2


# ======================================================================
# LÍMITES DE LOS GENES (18 parámetros optimizables)
# ======================================================================
GENE_BOUNDS = {
    'KP_GIRO':                 (0.0, 1.9),     # 0.9 +- 1
    'KI_GIRO':                 (0.0, 1.08),    # 0.08 +- 1
    'KD_GIRO':                 (0.0, 1.8),     # 0.8 +- 1
    'VEL_BASE':                (0.0, 3.0),     # 1.0 +- 2
    'VEL_LEJOS':               (4.0, 8.0),     # 6.0 +- 2
    'ALPHA_VEL':               (0.0, 1.0),     # 0.4 +- 1 (limitado 0-1)
    'OBS_FRONT_NEAR':          (0.01, 1.2),    # 0.20 +- 1 (min 0.01)
    'OBS_FRONT_HARD':          (0.01, 1.1),    # 0.10 +- 1
    'OBS_SIDE_NEAR':           (0.01, 1.08),   # 0.08 +- 1
    'AVOID_BACK_SPEED_STRONG': (-4.8, -0.8),   # -2.8 +- 2
    'AVOID_BACK_SPEED_SOFT':   (-1.2, 2.8),    # 0.8 +- 2
    'AVOID_BACK_STEPS':        (4.0, 6.0),     # 5.0 +- 1
    'AVOID_CLEAR_SPEED_FAST':  (3.0, 7.0),     # 5.0 +- 2
    'AVOID_CLEAR_SPEED_SLOW':  (1.0, 5.0),     # 3.0 +- 2
    'AVOID_CLEAR_STEPS':       (11.0, 13.0),   # 12.0 +- 1
    'STUCK_AVOID_LIMIT':       (2.0, 4.0),     # 3.0 +- 1
    'ESCAPE_BACK_SPEED':       (-5.2, -1.2),   # -3.2 +- 2
    'ESCAPE_BACK_STEPS':       (7.0, 9.0),     # 8.0 +- 1
}

# ======================================================================
# POSICIONES DE SPAWN (ajustar según tu escena .ttt)
# ======================================================================
SPAWN_POSES = [
    (+0.23053, -0.950, +0.13879, 0.0, 0.0, -30.00),
    (+3.08053, +1.275, +0.13879, 0.0, 0.0, -155.00),
    (+3.08053, -0.925, +0.13879, 0.0, 0.0, +135.00)
]

# ======================================================================
# HIPERPARÁMETROS DEL GA
# ======================================================================
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.35
ELITE_COUNT = 2              
TOURNAMENT_SIZE = 4          
MAX_SIMULATION_TIME = 120.0  



# ======================================================================
# GENOMA
# ======================================================================
class Genome:
    def __init__(self, config=None):
        if config is None:
            self.config = copy.deepcopy(DEFAULT_CONFIG)
            for k, bounds in GENE_BOUNDS.items():
                self.config[k] = random.uniform(bounds[0], bounds[1])
        else:
            self.config = copy.deepcopy(config)
        self.fitness = 0.0

    def mutate(self):
        for k, bounds in GENE_BOUNDS.items():
            if random.random() < MUTATION_RATE:
                rango = bounds[1] - bounds[0]
                mutation = random.gauss(0, rango * 0.08)
                new_val = self.config[k] + mutation
                self.config[k] = max(bounds[0], min(bounds[1], new_val))


def crossover(parent1, parent2):
    child_config = copy.deepcopy(parent1.config)
    for k in GENE_BOUNDS.keys():
        if random.random() < 0.5:
            child_config[k] = parent2.config[k]
    return Genome(child_config)


def tournament_select(population, k=TOURNAMENT_SIZE):
    contenders = random.sample(population, min(k, len(population)))
    return max(contenders, key=lambda g: g.fitness)


# ======================================================================
# RESET DE ESTADO GLOBAL DE modelo_final
# ======================================================================
def reset_modulo_estado():
    m2._prev_vel = (0.0, 0.0)
    m2._integral_x = 0.0
    m2._prev_err_x = 0.0
    m2._last_seen_side = 0
    m2._lost_counter = m2.SEARCH_MEMORY_STEPS
    m2._avoid_dir = 0
    m2._avoid_phase = "none"
    m2._avoid_steps = 0
    m2._consecutive_avoids = 0
    m2._last_mode = "init"


# ======================================================================
# EVALUACIÓN DE UN INDIVIDUO
# ======================================================================
def evaluate_individual(genome, coppelia, robot, robot_handle, spawn_pose):
    print(f"\n>> Evaluando: KP={genome.config['KP_GIRO']:.3f}, KI={genome.config['KI_GIRO']:.3f}, KD={genome.config['KD_GIRO']:.3f}")

    coppelia.stop_simulation()
    time.sleep(0.5)
    coppelia.start_simulation()
    time.sleep(1.0)

    x, y, z, alpha, beta, gamma = spawn_pose
    try:
        coppelia.sim.setObjectPosition(robot_handle, [x, y, z], coppelia.sim.handle_world)
        coppelia.sim.setObjectOrientation(robot_handle,
                                          [math.radians(alpha),
                                           math.radians(beta),
                                           math.radians(gamma)],
                                          coppelia.sim.handle_world)
        time.sleep(0.3)
    except Exception as e:
        print(f"Warning: Error reposicionando robot: {e}")

    reset_modulo_estado()
    config = genome.config

    total_forward_dist = 0.0
    ball_tracking_error = 0.0
    crashed = False

    start_time = coppelia.sim.getSimulationTime()
    last_time = start_time

    try:
        while True:
            if not coppelia.is_running():
                break

            now = coppelia.sim.getSimulationTime()
            current_sim_time = now - start_time
            if current_sim_time >= MAX_SIMULATION_TIME:
                break

            sonars = robot.get_sonar()
            if not sonars or len(sonars) < 16:
                time.sleep(0.01)
                continue
                
            img = robot.get_image()
            
            dt = now - last_time
            if dt <= 0:
                dt = 0.02
            last_time = now

            left_speed, right_speed, mode, err_x, deteccion, obs, mask = m2.step_v1(
                sonars, img, config
            )
            
            robot.set_speed(left_speed, right_speed)

            if min(sonars[:8]) < 0.08:
                crashed = True
                break

            forward_speed = 0.5 * (left_speed + right_speed)
            if forward_speed > 0:
                total_forward_dist += forward_speed * dt

            # Cálculo de penalización de persecución (análogo a la pared)
            if mode == "track" or mode == "track_after_avoid":
                error = abs(err_x)
                if error > 0.15:
                    ball_tracking_error += (error ** 2) * dt * 25.0
                else:
                    ball_tracking_error += error * dt
            else:
                ball_tracking_error += dt * 1.5

            time.sleep(0.02)

    except Exception as e:
        print(f"Excepción en la simulación: {e}")
        crashed = True

    finally:
        robot.set_speed(0, 0)
        coppelia.stop_simulation()
        time.sleep(0.5)

    time_penalty = MAX_SIMULATION_TIME * 1.5
    genome.fitness = total_forward_dist - (8.0 * ball_tracking_error) - time_penalty
    
    if crashed:
        # Penalización si choca (se asume que fitness será multiplicada si es negativa, o le restamos para asegurar)
        if genome.fitness > 0:
            genome.fitness = -genome.fitness * 100.0
        else:
            genome.fitness *= 100.0
            
        print(f"Colisión detectada (-100 pts): {total_forward_dist:.2f} | Castigo Bola: {ball_tracking_error:.2f} | Castigo T: {time_penalty:.2f} | Fitness: {genome.fitness:.2f}")
    else:
        print(f"Tiempo agotado. Dist: {total_forward_dist:.2f} | Castigo Bola: {ball_tracking_error:.2f} | Castigo T: {time_penalty:.2f} | Fitness: {genome.fitness:.2f}")

    return genome.fitness


# ======================================================================
# POBLACIÓN INICIAL
# ======================================================================
def init_population():
    pop = [Genome(DEFAULT_CONFIG)]
    for _ in range(POPULATION_SIZE - 1):
        pop.append(Genome())
    return pop


# ======================================================================
# MAIN
# ======================================================================
def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)
    robot_handle = coppelia.sim.getObject('/PioneerP3DX')

    population = init_population()
    current_pose_index = 0

    best_ever = None

    for gen in range(GENERATIONS):
        print(f"\n{'='*50}")
        print(f"  GENERACIÓN {gen+1}/{GENERATIONS}")
        print(f"  Pose de spawn: #{current_pose_index + 1} → {SPAWN_POSES[current_pose_index]}")
        print(f"{'='*50}")

        if gen > 0 and gen % 5 == 0:
            current_pose_index = (current_pose_index + 1) % len(SPAWN_POSES)
            print(f"\n  Cambiando a pose #{current_pose_index + 1}")

        current_pose = SPAWN_POSES[current_pose_index]

        for i, ind in enumerate(population):
            print(f"\n--- Individuo {i+1}/{POPULATION_SIZE} ---")
            evaluate_individual(ind, coppelia, robot, robot_handle, current_pose)

        population.sort(key=lambda x: x.fitness, reverse=True)

        print(f"\n  Mejor fitness gen {gen+1}: {population[0].fitness:.2f}")

        if best_ever is None or population[0].fitness > best_ever.fitness:
            best_ever = copy.deepcopy(population[0])

        if gen == GENERATIONS - 1:
            break

        next_pop = []
        for i in range(ELITE_COUNT):
            next_pop.append(copy.deepcopy(population[i]))

        while len(next_pop) < POPULATION_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = crossover(p1, p2)
            child.mutate()
            next_pop.append(child)

        population = next_pop

    print("\n\n" + "=" * 60)
    print("  OPTIMIZACIÓN GA FINALIZADA")
    print("=" * 60)
    print(f"\n  Mejor Fitness Global: {best_ever.fitness:.3f}")
    print("\n  Copia estos valores en DEFAULT_CONFIG de modelo_final.py:\n")

    print("  DEFAULT_CONFIG = {")
    for k in GENE_BOUNDS:
        print(f"      '{k}': {best_ever.config[k]:.4f},")
    print("  }")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
