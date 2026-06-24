import time
import random
import copy
import math
import robotica
from wall_follow_pid import State, PID, wall_follow_step, DEFAULT_CONFIG


GENE_BOUNDS = {
    'KP': (0.0, 5.0),
    'KI': (0.0, 1.0),
    'KD': (0.0, 5.0),
}

SPAWN_POSES = [
    (-1.76947, +0.125, +0.13879, 0.0, 0.0, -90),   
    (+0.93053, -2.200, +0.13879, 0.0, 0.0, +180),   
    (-2.19447, +1.775, +0.13879, 0.0, 0.0, +90.00),   
    (+0.93053, +0.175, +0.13879, 0.0, 0.0, +90),   
    (+1.00553, +1.225, +0.13879, 0.0, 0.0, 0.0),   
    (-1.76947, +0.125, +0.13879, 0.0, 0.0, -90),   
    (+0.93053, -2.200, +0.13879, 0.0, 0.0, +180),   
    (-2.19447, +1.775, +0.13879, 0.0, 0.0, +90.00),   
    (+0.93053, +0.175, +0.13879, 0.0, 0.0, +90),   
    (+1.00553, +1.225, +0.13879, 0.0, 0.0, 0.0)  
]

POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.35
MAX_SIMULATION_TIME = 200.0
LAP_RADIUS = 0.5


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


def evaluate_individual(genome, coppelia, robot, robot_handle, spawn_pose):
    print(f"\n>> Evaluando: KP={genome.config['KP']:.3f}, KI={genome.config['KI']:.3f}, KD={genome.config['KD']:.3f}")
    
    coppelia.stop_simulation()
    time.sleep(0.5)
    coppelia.start_simulation()
    time.sleep(1.0)
    
    x, y, z, alpha, beta, gamma = spawn_pose
    try:
        coppelia.sim.setObjectPosition(robot_handle, [x, y, z], coppelia.sim.handle_world)
        coppelia.sim.setObjectOrientation(robot_handle, [alpha, beta, gamma], coppelia.sim.handle_world)
        time.sleep(0.3)
    except Exception as e:
        print(f"Warning: Error reposicionando robot: {e}")

    try:
        start_pos = coppelia.sim.getObjectPosition(robot_handle, coppelia.sim.handle_world)
    except Exception as e:
        print(f"Warning: Error obteniendo posición inicial: {e}")
        start_pos = None
    
    pid = PID(
        kp=genome.config['KP'], 
        ki=genome.config['KI'], 
        kd=genome.config['KD']
    )
    state = State.SEARCH_WALL
    
    total_forward_dist = 0.0
    wall_tracking_error = 0.0
    crashed = False
    completed_lap = False
    lap_time = 0.0
    
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
                
            if current_sim_time >= 20.0 and start_pos is not None:
                current_pos = coppelia.sim.getObjectPosition(robot_handle, coppelia.sim.handle_world)
                dist_to_start = math.sqrt((current_pos[0] - start_pos[0])**2 + (current_pos[1] - start_pos[1])**2)
                if dist_to_start < LAP_RADIUS:
                    completed_lap = True
                    lap_time = current_sim_time
                    break
                    
            readings = robot.get_sonar()
            if not readings or len(readings) < 16:
                time.sleep(0.01)
                continue
                
            dt = now - last_time
            if dt <= 0: dt = 0.02
            last_time = now
            
            lspeed, rspeed, state, front, iz1, iz2, lat = wall_follow_step(
                readings, pid, state, dt, config=genome.config
            )
            robot.set_speed(lspeed, rspeed)
            
            if min(readings) < 0.08:
                crashed = True
                break
                
            forward_speed = 0.5 * (lspeed + rspeed)
            if forward_speed > 0:
                total_forward_dist += forward_speed * dt
                
            des_dist = genome.config['DESIRED_WALL_DISTANCE']
            error = abs(lat - des_dist)
            
            if state == State.SEARCH_WALL:
                wall_tracking_error += dt * 1.5
            else:
                if error > 0.15:
                    wall_tracking_error += (error ** 2) * dt * 25.0
                else:
                    wall_tracking_error += error * dt

            time.sleep(0.02)
            
    except Exception as e:
        print(f"Excepción en la simulación: {e}")
        crashed = True
        
    finally:
        robot.set_speed(0, 0)
        coppelia.stop_simulation()
        time.sleep(0.5)
    
    time_taken = lap_time if completed_lap else MAX_SIMULATION_TIME
    time_penalty = time_taken * 1.5
    genome.fitness = total_forward_dist - (8.0 * wall_tracking_error) - time_penalty
    if crashed:
        genome.fitness *= 100.0
        print(f"Colisión detectada (-100 pts): {total_forward_dist:.2f} | Castigo Pared: {wall_tracking_error:.2f} | Castigo T: {time_penalty:.2f} | Fitness: {genome.fitness:.2f}")
    else:
        if completed_lap:
            print(f"VUELTA COMPLETADA en {lap_time:.2f}s | Dist: {total_forward_dist:.2f} | Castigo Pared: {wall_tracking_error:.2f} | Castigo T: {time_penalty:.2f} | Fitness: {genome.fitness:.2f}")
        else:
            genome.fitness += -50.0
            print(f"Tiempo agotado. Dist: {total_forward_dist:.2f} | Castigo Pared: {wall_tracking_error:.2f} | Castigo T: {time_penalty:.2f} | Fitness: {genome.fitness:.2f}")

    return genome.fitness


def init_population():
    pop = [Genome(Config) for Config in [DEFAULT_CONFIG]]
    for _ in range(POPULATION_SIZE - 1):
        pop.append(Genome())
    return pop


def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')
    robot_handle = coppelia.sim.getObject('/PioneerP3DX')
    
    population = init_population()
    current_pose_index = 0  
    
    for gen in range(GENERATIONS):
        print(f"\n================================")
        print(f"  GENERACIÓN {gen+1}/{GENERATIONS}  ")
        print(f"  Pose de spawn: #{current_pose_index + 1} → {SPAWN_POSES[current_pose_index]}")
        print(f"================================")
        
        if gen > 0 and gen % 5 == 0:
            current_pose_index = (current_pose_index + 1) % len(SPAWN_POSES)
            print(f"\nCambiando a pose de spawn #{current_pose_index + 1}: {SPAWN_POSES[current_pose_index]}")
        
        current_pose = SPAWN_POSES[current_pose_index]
        
        for ind in population:
            evaluate_individual(ind, coppelia, robot, robot_handle, current_pose)
            
        population.sort(key=lambda x: x.fitness, reverse=True)
        print(f"\nMejor fitness de la Generación {gen+1}: {population[0].fitness:.2f}")
        print(f"KP={population[0].config['KP']:.4f}, KI={population[0].config['KI']:.4f}, KD={population[0].config['KD']:.4f}")
        
        if gen == GENERATIONS - 1:
            break
            
        next_pop = [population[0], population[1]]
        
        while len(next_pop) < POPULATION_SIZE:
            p1 = random.choice(population[:4]) 
            p2 = random.choice(population[:4])
            child = crossover(p1, p2)
            child.mutate()
            next_pop.append(child)
            
        population = next_pop

    best = population[0]
    print("\n\n" + "[]" * 20)
    print("AFINAMIENTO DE PARÁMETROS PID FINALIZADO")
    print(f"Mejor Fitness Puntuado: {best.fitness:.3f}")
    print("Copia y pega estos valores PID en tu DEFAULT_CONFIG de wall_follow_pid_fsm_adapted:\n")
    for k in ['KP', 'KI', 'KD']:
        print(f"    '{k}': {best.config[k]:.4f},")
    print("[]" * 20)


if __name__ == '__main__':
    main()
