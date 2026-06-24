import time
from enum import Enum, auto

import robotica


# =========================
# CONFIGURACIÓN GENERAL
# =========================
DEFAULT_CONFIG = {
    'DESIRED_WALL_DISTANCE': 0.35,
    'THRESH_IZQ1_0': 0.2,
    'THRESH_IZQ1_1': 0.3,
    'THRESH_IZQ2_15': 0.3,
    'THRESH_IZQ2_14': 0.4,
    'THRESH_FRONTAL': 0.4,
    'THRESH_FRONTAL_LAT_DER': 0.5,
    'THRESH_LATERAL': 0.25,
    'FRONT_CLEAR_DISTANCE': 0.42,
    'BASE_SPEED': 1.5,
    'TURN_SPEED_POS': 1.5,
    'TURN_SPEED_NEG': -0.4,
    'MAX_SPEED': 5,
    'ANGLE_WEIGHT': 1.2,
    'KP': 1.5,
    'KI': 0.08,
    'KD': 0.9
}


# =========================
# MAQUINA DE ESTADOS
# =========================
class State(Enum):
    SEARCH_WALL = auto()
    FOLLOW_WALL = auto()
    TURN_TO_WALL = auto()
    AVOID_FRONT = auto()


# =========================
# PID
# =========================
class PID:
    def __init__(self, kp, ki, kd, out_min=-1.2, out_max=1.2, integral_limit=1.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_derivative = 0.0
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_derivative = 0.0
        self.initialized = False

    def update(self, error, dt):
        if dt <= 1e-6:
            return 0.0

        if not self.initialized:
            self.prev_error = error
            self.initialized = True

        self.integral += error * dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        raw_derivative = (error - self.prev_error) / dt
        derivative = 0.6 * self.last_derivative + 0.4 * raw_derivative
        self.last_derivative = derivative
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(u, self.out_min, self.out_max)


# =========================
# UTILS
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sonar_groups(readings):
    front = min([readings[3], readings[4], readings[5], readings[6]])
    iz1 = min(readings[0], readings[1])    
    iz2 = min(readings[14], readings[15])  
    lat = readings[2]                      
    return front, iz1, iz2, lat


def wall_follow_step(readings, pid, state, dt, config=None):
    if config is None:
        config = DEFAULT_CONFIG

    thresh_izq1_0 = config.get('THRESH_IZQ1_0', 0.2)
    thresh_izq1_1 = config.get('THRESH_IZQ1_1', 0.3)
    thresh_izq2_15 = config.get('THRESH_IZQ2_15', 0.3)
    thresh_izq2_14 = config.get('THRESH_IZQ2_14', 0.4)
    thresh_frontal = config.get('THRESH_FRONTAL', 0.4)
    thresh_lateral = config.get('THRESH_LATERAL', 0.3)
    
    cond_1 = (readings[0] < thresh_izq1_0 or readings[1] < thresh_izq1_1)
    cond_2 = (readings[15] < thresh_izq2_15 or readings[14] < thresh_izq2_14)
    frontal = (min(readings[3], readings[4]) < thresh_frontal)
    lateral_cond = (readings[2] < thresh_lateral)

    lat_dist = readings[2]
    iz1_dist = min(readings[0], readings[1])
    iz2_dist = min(readings[15], readings[14])


    new_state = state
    if (cond_1 and frontal) or lateral_cond:
        new_state = State.AVOID_FRONT
        pid.reset()
        
        lspeed, rspeed = config.get('TURN_SPEED_POS', 0.8), config.get('TURN_SPEED_NEG', -0.2)
    
            
    elif (not cond_1 and cond_2 and not frontal):
        new_state = State.TURN_TO_WALL
        pid.reset()
        lspeed, rspeed = config.get('TURN_SPEED_NEG', -0.2), config.get('TURN_SPEED_POS', 0.8)
        
    else:
        if lat_dist > 0.7:
            new_state = State.SEARCH_WALL
            pid.reset()
            lspeed, rspeed = config.get('BASE_SPEED', 1.5), config.get('BASE_SPEED', 1.5)
        else:
            new_state = State.FOLLOW_WALL
            error_dist = config.get('DESIRED_WALL_DISTANCE', 0.3) - lat_dist
            error_angle = (iz2_dist - iz1_dist)
            
            error = error_dist + config.get('ANGLE_WEIGHT', 1.2) * error_angle
            correction = pid.update(error, dt)
            
            # Reducción suave proporcional si la pared frontal se empieza a aproximar a la vista 
            # (Sin alterar bruscamente la decisión until AVOID)
            front_dist = min([readings[3], readings[4], readings[5], readings[6]])
            slowdown = 0.9
            if front_dist < config.get('FRONT_CLEAR_DISTANCE', 0.42):
                slowdown = clamp((front_dist - 0.2) / 0.3, 0.4, 0.9)
                
            base = config.get('BASE_SPEED', 1.5) * slowdown
            lspeed, rspeed = base + correction, base - correction

    lspeed = clamp(lspeed, -config.get('MAX_SPEED', 2.0), config.get('MAX_SPEED', 2.0))
    rspeed = clamp(rspeed, -config.get('MAX_SPEED', 2.0), config.get('MAX_SPEED', 2.0))
    
    return lspeed, rspeed, new_state, min([readings[3], readings[4], readings[5], readings[6]]), iz1_dist, iz2_dist, lat_dist


# =========================
# CONTROL PRINCIPAL
# =========================
def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')

    pid = PID(kp=DEFAULT_CONFIG['KP'], ki=DEFAULT_CONFIG['KI'], kd=DEFAULT_CONFIG['KD'])
    state = State.SEARCH_WALL

    print("Iniciando wall following con PID + FSM...")
    coppelia.start_simulation()

    last_time = coppelia.sim.getSimulationTime()
    cycle = 0

    try:
        while coppelia.is_running():
            readings = robot.get_sonar()

            if readings and len(readings) >= 16:
                now = coppelia.sim.getSimulationTime()
                dt = now - last_time
                if dt <= 0:
                    dt = 0.05
                last_time = now

                lspeed, rspeed, state, front, iz1, iz2, lat = wall_follow_step(
                    readings, pid, state, dt, config=DEFAULT_CONFIG
                )

                robot.set_speed(lspeed, rspeed)

                cycle += 1
                if cycle % 20 == 0:
                    print(
                        f"state={state.name:12s} | "
                        f"front={front:.3f} | iz1={iz1:.3f} | "
                        f"iz2={iz2:.3f} | lat={lat:.3f} | "
                        f"L={lspeed:.2f} R={rspeed:.2f}"
                    )

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Interrupción del usuario.")

    finally:
        robot.set_speed(0.0, 0.0)
        coppelia.stop_simulation()


if __name__ == '__main__':
    main()