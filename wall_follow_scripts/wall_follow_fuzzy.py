import time
from enum import Enum, auto
import robotica


# =========================================================
# CONFIGURACIÓN Y ESTADOS
# =========================================================
class State(Enum):
    FOLLOW_WALL = auto()
    AVOID_FRONT = auto()
    TURN_TO_WALL = auto()  

CONFIG = {
    'D_REF': 0.35,
    'BASE_SPEED': 1.5,
    'TURN_SPEED_POS': 1.5,
    'TURN_SPEED_NEG': -0.4,
    'THRESH_IZQ1_0': 0.2,
    'THRESH_IZQ1_1': 0.3,
    'THRESH_IZQ2_15': 0.3,
    'THRESH_IZQ2_14': 0.4,
    'THRESH_FRONTAL': 0.4,
    'THRESH_LATERAL': 0.25,
    'ANGLE_WEIGHT': 1.2,
    'FRONT_CLEAR_DISTANCE': 0.42
}

# =========================================================
# MOTOR BORROSO (Núcleo)
# =========================================================
def clamp(x, lo, hi): return max(lo, min(hi, x))

def tri(x, a, b, c):
    if x <= a or x >= c: return 0.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)

def trap(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if b <= x <= d: return 1.0 if x <= c else (d - x) / (d - c)
    return (x - a) / (b - a)

def fuzzy_engine(e_dist, e_ori):
    """
    e_dist: Error de distancia (D_REF - d_lat)  positivo=muy cerca, negativo=muy lejos
    e_ori:  Error de orientación (d_rear - d_front)  positivo=apuntando a pared, negativo=alejandose
    """
    d = {
        'CERCA': trap(e_dist, 0.01, 0.06, 0.5, 0.5),
        'OK':    tri(e_dist, -0.08, 0.0, 0.08),
        'LEJOS': trap(e_dist, -0.5, -0.5, -0.06, -0.01)
    }
    o = {
        'HACIA': trap(e_ori, 0.02, 0.05, 0.3, 0.3),
        'PARA':  tri(e_ori, -0.04, 0.0, 0.04),
        'ALEJA': trap(e_ori, -0.3, -0.3, -0.05, -0.02)
    }

    rules = {
        'MN': max(min(d['CERCA'], o['HACIA']),
               min(d['CERCA'], o['PARA'])),
        'N':  max(min(d['CERCA'], o['ALEJA']),
               min(d['OK'],    o['HACIA'])),
        'Z':  min(d['OK'], o['PARA']),
        'P':  max(min(d['LEJOS'], o['HACIA']),
               min(d['OK'],    o['ALEJA'])),
        'MP': max(min(d['LEJOS'], o['ALEJA']),
               min(d['LEJOS'], o['PARA']))
    }
    centers = {'MN': -1.0, 'N': -0.4, 'Z': 0.0, 'P': 0.4, 'MP': 1.0}
    num, den = 0.0, 0.0
    for label, weight in rules.items():
        num += weight * centers[label]
        den += weight
   
    return num / den if den > 0 else 0.0

# =========================================================
# LÓGICA DE CONTROL (FSM robusta basada en avoid.py)
# =========================================================
def get_navigation_data(readings):
    """Extrae datos de sensores con la misma agrupación que el PID."""
    # Front incluye sensores diagonales derechos para detectar mejor
    front = min([readings[3], readings[4]])
    d_lat1 = min(readings[0], readings[1])
    d_lat2 = min(readings[14], readings[15])
    d_diag = readings[2]
    
    return front, d_lat1, d_lat2, d_diag

def compute_state_and_speeds(readings, front, d_lat1, d_lat2, d_diag, config):
    cond_1 = (readings[0] < config['THRESH_IZQ1_0'] or
                readings[1] < config['THRESH_IZQ1_1'])
    cond_2 = (readings[15] < config['THRESH_IZQ2_15'] or
                readings[14] < config['THRESH_IZQ2_14'])
    frontal = min(readings[3], readings[4]) < config['THRESH_FRONTAL']
    lateral_cond = readings[2] < config['THRESH_LATERAL']
    
    if (cond_1 and frontal) or lateral_cond:
        state = State.AVOID_FRONT
        lspeed = config['TURN_SPEED_POS']
        rspeed = config['TURN_SPEED_NEG']

    elif not cond_1 and cond_2 and not frontal:
        state = State.TURN_TO_WALL
        lspeed = config['TURN_SPEED_NEG']
        rspeed = config['TURN_SPEED_POS']

    else:
        state = State.FOLLOW_WALL
        
        e_dist = config['D_REF'] - d_diag
        e_ori = (d_lat2 - d_lat1) * config['ANGLE_WEIGHT']
        
        turn_fuzzy = fuzzy_engine(e_dist, e_ori)
        
        slowdown = 0.9
        if front < config['FRONT_CLEAR_DISTANCE']:
            slowdown = clamp((front - 0.2) / 0.3, 0.4, 0.9)
        
        base = config['BASE_SPEED'] * slowdown
       
        lspeed, rspeed = base + turn_fuzzy, base - turn_fuzzy

    lspeed = clamp(lspeed, -3.0, 3.0)
    rspeed = clamp(rspeed, -3.0, 3.0)
    return state, lspeed, rspeed


def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')
    state = State.FOLLOW_WALL
    coppelia.start_simulation()
    cycle = 0

    try:
        while coppelia.is_running():
            readings = robot.get_sonar()
            if not readings or len(readings) < 16: continue

            front, d_f_diag, d_r_diag, d_lat = get_navigation_data(readings)
           
            state, lspeed, rspeed = compute_state_and_speeds(
                readings, front, d_f_diag, d_r_diag, d_lat, CONFIG
            )

            robot.set_speed(lspeed, rspeed)
            
            cycle += 1
            if cycle % 20 == 0:
                print(
                    f"state={state.name:12s} | "
                    f"front={front:.3f} | diag_f={d_f_diag:.3f} | "
                    f"diag_r={d_r_diag:.3f} | lat={d_lat:.3f} | "
                    f"L={lspeed:.2f} R={rspeed:.2f}"
                )
            time.sleep(0.05)

    finally:
        robot.set_speed(0, 0)
        coppelia.stop_simulation()

if __name__ == '__main__':
    main()