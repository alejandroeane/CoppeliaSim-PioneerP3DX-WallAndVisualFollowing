import cv2
import numpy as np
import robotica


# ======================================================================
# CONFIGURACIÓN POR DEFECTO (Ajustable por el GA)
# ======================================================================
DEFAULT_CONFIG = {
    'KP_GIRO': 0.7120,
    'KI_GIRO': 0.1964,
    'KD_GIRO': 0.7905,

    'VEL_BASE': 0.7534,
    'VEL_LEJOS': 5.0764,
    'ALPHA_VEL': 0.8408,

    'OBS_FRONT_NEAR': 0.152,
    'OBS_FRONT_HARD': 0.0761,
    'OBS_SIDE_NEAR': 0.0100,

    'AVOID_BACK_SPEED_STRONG': -2.9566,
    'AVOID_BACK_SPEED_SOFT': 0.9093,
    'AVOID_BACK_STEPS': 5.3272,

    'AVOID_CLEAR_SPEED_FAST': 5.8015,
    'AVOID_CLEAR_SPEED_SLOW': 3.6224,
    'AVOID_CLEAR_STEPS': 12.7042,

    'STUCK_AVOID_LIMIT': 2.1255,
    'ESCAPE_BACK_SPEED': -4.3339,
    'ESCAPE_BACK_STEPS': 7.2292


    # # PID visual
    # 'KP_GIRO': 0.9,
    # 'KI_GIRO': 0.08,
    # 'KD_GIRO': 0.8,
    
    # # Velocidades de persecución
    # 'VEL_BASE': 1.0,
    # 'VEL_LEJOS': 6.0,
    # 'ALPHA_VEL': 0.4,
    
    # # Umbrales de detección de obstáculos
    # 'OBS_FRONT_NEAR': 0.20,
    # 'OBS_FRONT_HARD': 0.10,
    # 'OBS_SIDE_NEAR': 0.08,
    
    # # Maniobras de evasión: BACK
    # 'AVOID_BACK_SPEED_STRONG': -2.8,
    # 'AVOID_BACK_SPEED_SOFT': 0.8,
    # 'AVOID_BACK_STEPS': 5.0,
    
    # # Maniobras de evasión: CLEAR
    # 'AVOID_CLEAR_SPEED_FAST': 5.0,
    # 'AVOID_CLEAR_SPEED_SLOW': 3.0,
    # 'AVOID_CLEAR_STEPS': 12.0,
    
    # # Maniobras de evasión: ESCAPE
    # 'STUCK_AVOID_LIMIT': 3.0,
    # 'ESCAPE_BACK_SPEED': -3.2,
    # 'ESCAPE_BACK_STEPS': 8.0,
}

# ======================================================================
# PARÁMETROS FIJOS
# ======================================================================
CX_REF = 128
AREA_MIN = 120.0
AREA_CERCA = 60000.0
AREA_LEJOS = 5000.0
VEL_MAX = 8.0

SEARCH_TURN_FAST = 1.0
SEARCH_TURN_MEMORY = 1.5
SEARCH_MEMORY_STEPS = 25
SEARCH_FWD_GAIN = 0.45
SEARCH_FWD_MIN = 0.0
SEARCH_FWD_MAX = 1.8

# ======================================================================
# ESTADO PERSISTENTE
# ======================================================================
_prev_vel = (0.0, 0.0)
_integral_x = 0.0
_prev_err_x = 0.0

_last_seen_side = 0
_lost_counter = 0

_avoid_dir = 0
_avoid_phase = "none"
_avoid_steps = 0
_consecutive_avoids = 0
_last_mode = "init"


# ======================================================================
# UTILIDADES
# ======================================================================
def saturar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def reset_control_visual():
    global _prev_vel, _integral_x, _prev_err_x
    _prev_vel = (0.0, 0.0)
    _integral_x = 0.0
    _prev_err_x = 0.0


# ======================================================================
# VISIÓN
# ======================================================================
def filtro_umbralizacion(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    bajo1 = np.array([0, 80, 80], dtype=np.uint8)
    alto1 = np.array([10, 255, 255], dtype=np.uint8)
    bajo2 = np.array([160, 80, 80], dtype=np.uint8)
    alto2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, bajo1, alto1)
    mask2 = cv2.inRange(hsv, bajo2, alto2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detectar_bola(mask, img_rgb=None):
    if img_rgb is None:
        return None
        
    mask_suavizada = cv2.GaussianBlur(mask, (9, 9), 2)
    circulos = cv2.HoughCircles(
        mask_suavizada, cv2.HOUGH_GRADIENT, dp=1, minDist=50,     
        param1=70, param2=20, minRadius=5, maxRadius=220     
    )
    
    if circulos is None:
        return None
        
    circulos = np.uint16(np.around(circulos))[0, :]
    candidato_valido = None
    max_area = 0
    
    for c in circulos:
        cx, cy, radio = c[0], c[1], c[2]
        if cx < 0 or cx >= mask.shape[1] or cy < 0 or cy >= mask.shape[0]:
            continue
            
        if mask[cy, cx] > 0:
            area = float(np.pi * (radio ** 2))
            if area > max_area and area > AREA_MIN:
                max_area = area
                candidato_valido = (radio, area, cx, cy)
                
    return candidato_valido


def actualizar_memoria_bola(cx):
    global _last_seen_side, _lost_counter, _consecutive_avoids
    if cx < CX_REF - 10:
        _last_seen_side = -1
    elif cx > CX_REF + 10:
        _last_seen_side = 1
    else:
        _last_seen_side = 0

    _lost_counter = 0
    _consecutive_avoids = 0


# ======================================================================
# PERSECUCIÓN VISUAL
# ======================================================================
def control_visual(cx, area, config):
    global _prev_vel, _integral_x, _prev_err_x

    error_x = (cx - CX_REF) / CX_REF
    _integral_x += error_x
    _integral_x = np.clip(_integral_x, -5.0, 5.0)
    derivativa_x = error_x - _prev_err_x
    _prev_err_x = error_x

    giro = (config['KP_GIRO'] * error_x + 
            config['KI_GIRO'] * _integral_x + 
            config['KD_GIRO'] * derivativa_x)

    t = np.clip((area - AREA_LEJOS) / (AREA_CERCA - AREA_LEJOS), 0.0, 1.0)
    vel_fwd = config['VEL_LEJOS'] + t * (config['VEL_BASE'] - config['VEL_LEJOS'])

    factor_centrado = max(0.55, 1.0 - 0.35 * abs(error_x))
    avance = vel_fwd * factor_centrado

    raw_izq = avance + giro
    raw_der = avance - giro
    raw_izq = np.clip(raw_izq, -VEL_MAX, VEL_MAX)
    raw_der = np.clip(raw_der, -VEL_MAX, VEL_MAX)

    alpha = config['ALPHA_VEL']
    left_speed = _prev_vel[0] + alpha * (raw_izq - _prev_vel[0])
    right_speed = _prev_vel[1] + alpha * (raw_der - _prev_vel[1])
    _prev_vel = (left_speed, right_speed)

    return float(left_speed), float(right_speed), float(error_x)


# ======================================================================
# BÚSQUEDA
# ======================================================================
def control_busqueda(config):
    global _lost_counter, _prev_vel, _integral_x, _prev_err_x

    if _lost_counter < SEARCH_MEMORY_STEPS:
        _lost_counter += 1

        prev_fwd = 0.5 * (_prev_vel[0] + _prev_vel[1])
        search_fwd = SEARCH_FWD_GAIN * prev_fwd
        search_fwd = saturar(search_fwd, SEARCH_FWD_MIN, SEARCH_FWD_MAX)

        giro = config['KP_GIRO'] * _prev_err_x + config['KI_GIRO'] * _integral_x
        
        left_speed = search_fwd + giro
        right_speed = search_fwd - giro
        return left_speed, right_speed, "search_memory"

    reset_control_visual()
    return -SEARCH_TURN_FAST, SEARCH_TURN_FAST, "search_spin"


# ======================================================================
# SENSORES DE OBSTÁCULOS
# ======================================================================
def leer_obstaculos(sonars):
    front_left = min(sonars[1], sonars[2], sonars[3])
    front_right = min(sonars[6], sonars[5], sonars[4])
    side_left = sonars[0]
    side_right = sonars[7]
    front = min(front_left, front_right)

    return {
        "front": front,
        "front_left": front_left,
        "front_right": front_right,
        "side_left": side_left,
        "side_right": side_right,
    }


def detectar_obstaculo(obs, config):
    if obs["front"] < config['OBS_FRONT_NEAR']:
        return True
    if obs["side_left"] < config['OBS_SIDE_NEAR']:
        return True
    if obs["side_right"] < config['OBS_SIDE_NEAR']:
        return True
    return False


# ======================================================================
# EVITACIÓN DE OBSTÁCULOS (Máquina de estados v1)
# ======================================================================
def iniciar_evitacion(obs, config):
    global _avoid_dir, _avoid_phase, _avoid_steps, _lost_counter, _consecutive_avoids

    _consecutive_avoids += 1

    if _consecutive_avoids >= int(config['STUCK_AVOID_LIMIT']):
        _avoid_phase = "escape"
        _avoid_steps = int(config['ESCAPE_BACK_STEPS'])
        _lost_counter = SEARCH_MEMORY_STEPS
        reset_control_visual()
        return

    if obs["front_right"] <= obs["front_left"] or obs["side_right"] < config['OBS_SIDE_NEAR']:
        _avoid_dir = -1
    else:
        _avoid_dir = 1

    _avoid_phase = "back"
    _avoid_steps = int(config['AVOID_BACK_STEPS'])
    _lost_counter = SEARCH_MEMORY_STEPS
    reset_control_visual()


def control_evitacion(obs, config):
    global _avoid_phase, _avoid_steps

    if _avoid_phase == "escape":
        if _avoid_steps > 0:
            _avoid_steps -= 1
            spd = config['ESCAPE_BACK_SPEED']
            return spd, spd, "escape_back"
        _avoid_phase = "none"
        return None

    if _avoid_phase == "back":
        if _avoid_steps > 0 and (obs["front"] < config['OBS_FRONT_NEAR'] or obs["front"] < config['OBS_FRONT_HARD']):
            _avoid_steps -= 1
            if _avoid_dir == -1:
                return config['AVOID_BACK_SPEED_STRONG'], config['AVOID_BACK_SPEED_SOFT'], "avoid_back_from_right"
            return config['AVOID_BACK_SPEED_SOFT'], config['AVOID_BACK_SPEED_STRONG'], "avoid_back_from_left"
            
        _avoid_phase = "clear"
        _avoid_steps = int(config['AVOID_CLEAR_STEPS'])

    if _avoid_phase == "clear":
        if _avoid_steps > 0:
            _avoid_steps -= 1
            if _avoid_dir == -1:
                return config['AVOID_CLEAR_SPEED_FAST'], config['AVOID_CLEAR_SPEED_SLOW'], "avoid_clear_right"
            return config['AVOID_CLEAR_SPEED_SLOW'], config['AVOID_CLEAR_SPEED_FAST'], "avoid_clear_left"
            
        _avoid_phase = "none"
        return None

    return None


# ======================================================================
# ENCAPSULACIÓN PARA EL GA
# ======================================================================
def step_v1(sonars, img_rgb, config):
    """
    Da un paso en la simulación usando la lógica de máquina de estados v1.
    Retorna: (left_speed, right_speed, mode, error_x_visual)
    """
    global _avoid_phase

    obs = leer_obstaculos(sonars)
    mask = filtro_umbralizacion(img_rgb)
    deteccion = detectar_bola(mask, img_rgb)
    
    error_x_visual = 0.0

    if _avoid_phase != "none":
        evit = control_evitacion(obs, config)
        if evit is not None:
            left_speed, right_speed, mode = evit
        else:
            if deteccion is not None:
                _, area, cx, _ = deteccion
                actualizar_memoria_bola(cx)
                left_speed, right_speed, error_x_visual = control_visual(cx, area, config)
                mode = "track"
            else:
                left_speed, right_speed, mode = control_busqueda(config)

    elif detectar_obstaculo(obs, config):
        iniciar_evitacion(obs, config)
        evit = control_evitacion(obs, config)
        if evit is not None:
            left_speed, right_speed, mode = evit
        else:
            left_speed, right_speed, mode = control_busqueda(config)

    elif deteccion is not None:
        _, area, cx, _ = deteccion
        actualizar_memoria_bola(cx)
        left_speed, right_speed, error_x_visual = control_visual(cx, area, config)
        mode = "track"

    else:
        left_speed, right_speed, mode = control_busqueda(config)

    left_speed = saturar(left_speed, -VEL_MAX, VEL_MAX)
    right_speed = saturar(right_speed, -VEL_MAX, VEL_MAX)

    return left_speed, right_speed, mode, error_x_visual, deteccion, obs, mask


# ======================================================================
# DEBUG
# ======================================================================
def dibujar_debug(img_rgb, mask, deteccion, obs, mode):
    debug = img_rgb.copy()
    h, _ = debug.shape[:2]
    cv2.line(debug, (CX_REF, 0), (CX_REF, h - 1), (255, 255, 0), 1)

    mask_suavizada = cv2.GaussianBlur(mask, (9, 9), 2)
    circulos = cv2.HoughCircles(
        mask_suavizada, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=70, param2=20, minRadius=5, maxRadius=220
    )
    
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for i in circulos[0, :]:
            centro = (i[0], i[1])
            radio = i[2]
            cv2.circle(debug, centro, radio, (0, 255, 0), 2)

    if deteccion is not None:
        radio, area, cx, cy = deteccion
        cv2.circle(debug, (cx, cy), radio, (255, 0, 255), 2)
        cv2.circle(debug, (cx, cy), 5, (255, 0, 0), -1)

    y0 = 46
    dy = 20
    cv2.putText(debug, f"mode: {mode}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.putText(debug, f"lost={_lost_counter} side={_last_seen_side}", (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    
    if _avoid_phase != "none":
        cv2.putText(debug, f"avoid: {_avoid_phase} dir={_avoid_dir} n={_consecutive_avoids}",
                    (10, y0 + 2 * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

    if obs is not None:
        cv2.putText(debug, f"front={obs['front']:.2f} sL={obs['side_left']:.2f} sR={obs['side_right']:.2f}",
                    (10, y0 + 3 * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

    return debug


# ======================================================================
# MAIN
# ======================================================================
def main(args=None):
    global _prev_vel, _integral_x, _prev_err_x
    global _last_seen_side, _lost_counter
    global _avoid_dir, _avoid_phase, _avoid_steps, _consecutive_avoids, _last_mode

    _prev_vel = (0.0, 0.0)
    _integral_x = 0.0
    _prev_err_x = 0.0
    _last_seen_side = 0
    _lost_counter = SEARCH_MEMORY_STEPS
    _avoid_dir = 0
    _avoid_phase = "none"
    _avoid_steps = 0
    _consecutive_avoids = 0
    _last_mode = "init"

    config = DEFAULT_CONFIG.copy()
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)

    print("Iniciando control v1 encapsulado...")
    coppelia.start_simulation()

    try:
        while coppelia.is_running():
            sonars = robot.get_sonar()
            img = robot.get_image()

            left_speed, right_speed, mode, err_x, deteccion, obs, mask = step_v1(sonars, img, config)
            
            _last_mode = mode
            robot.set_speed(left_speed, right_speed)

            debug = dibujar_debug(img, mask, deteccion, obs, mode)
            cv2.imshow('mascara', mask)
            cv2.imshow('debug', debug)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        robot.set_speed(0.0, 0.0)
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
