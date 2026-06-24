import cv2
import numpy as np
import robotica


# ======================================================================
# PARÁMETROS DEL CONTROLADOR VISUAL / PERSECUCIÓN
# ======================================================================
CX_REF = 128             # Centro deseado de la imagen.
AREA_MIN = 120.0         # Filtra ruido pequeño.

KP_GIRO    = 0.9         # Ganancia proporcional angular.
KI_GIRO    = 0.08        # Ganancia integral (reducida para evitar windup).
KD_GIRO    = 0.8         # Ganancia derivativa.

# Umbrales de área ajustados a valores REALES observados (150–3500)
AREA_CERCA = 60000.0      # area >= esto → pelota cerca → VEL_BASE
AREA_LEJOS = 5000.0       # area <= esto → pelota muy lejos → VEL_LEJOS

VEL_BASE   = 1.0         # Velocidad de avance cuando la pelota está CERCA.
VEL_LEJOS  = 6.0         # Velocidad de avance cuando la pelota está LEJOS.
VEL_MAX    = 8.0         # Velocidad máxima de cada rueda.

# Suavizado exponencial (0..1, más alto = respuesta más rápida)
ALPHA_VEL  = 0.4

# ======================================================================
# PARÁMETROS DE BÚSQUEDA
# ======================================================================
SEARCH_TURN_FAST = 1.0          # Velocidad de giro cuando no hay memoria útil.
SEARCH_TURN_MEMORY = 1.5        # Velocidad de giro hacia el último lado donde se vio la bola.
SEARCH_MEMORY_STEPS = 25        # Iteraciones máximas manteniendo esa memoria.
SEARCH_FWD_GAIN = 0.45          # Porcentaje de la velocidad lineal previa que se conserva.
SEARCH_FWD_MIN = 0.0            # Mínimo avance durante búsqueda por memoria.
SEARCH_FWD_MAX = 1.8            # Máximo avance durante búsqueda por memoria.

# ======================================================================
# PARÁMETROS DE EVITACIÓN
# ======================================================================
OBS_FRONT_NEAR = 0.3           # Distancia frontal para activar evitación.
OBS_FRONT_HARD = 0.15           # Distancia frontal de peligro fuerte.
OBS_SIDE_NEAR = 0.08            # Distancia lateral para considerar obstáculo cercano.

AVOID_BACK_SPEED_STRONG = -2.8  # Rueda exterior del primer arco hacia atrás.
AVOID_BACK_SPEED_SOFT = 0.8     # Rueda interior del primer arco hacia atrás.
AVOID_BACK_STEPS = 5            # Duración del primer arco de retroceso.

AVOID_CLEAR_SPEED_FAST = 4.4    # Rueda exterior del segundo arco de despeje.
AVOID_CLEAR_SPEED_SLOW = 2.8    # Rueda interior del segundo arco de despeje.
AVOID_CLEAR_STEPS = 16          # Duración del arco de despeje.

STUCK_AVOID_LIMIT = 3           # Nº de entradas seguidas en evitación antes de asumir atasco.
ESCAPE_BACK_SPEED = -3.2        # Retroceso recto y fuerte para salir de esquinas/bloqueos.
ESCAPE_BACK_STEPS = 8           # Duración del escape de emergencia.

# ======================================================================
# ESTADO PERSISTENTE
# ======================================================================
_prev_vel = (0.0, 0.0)          # Últimas velocidades de rueda (para suavizado y memoria).
_integral_x = 0.0
_prev_err_x = 0.0

_last_seen_side = 0             # -1 izquierda, +1 derecha, 0 centrado/desconocido.
_lost_counter = 0               # Cuántos ciclos lleva sin ver la bola.

_avoid_dir = 0                  # -1: evitar hacia la derecha, +1: hacia la izquierda.
_avoid_phase = "none"           # none, back, clear, escape
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


def reset_pid_visual_only():
    global _integral_x, _prev_err_x
    _integral_x = 0.0
    _prev_err_x = 0.0


# ======================================================================
# VISIÓN
# ======================================================================
def filtro_umbralizacion(img_rgb):
    # Se mantiene COLOR_BGR2HSV porque en las pruebas esta conversión funcionaba bien.
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


def detectar_bola(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return None

    contorno = max(contornos, key=cv2.contourArea)
    momentos = cv2.moments(contorno)

    if momentos["m00"] <= AREA_MIN:
        return None

    cx = int(momentos["m10"] / momentos["m00"])
    cy = int(momentos["m01"] / momentos["m00"])
    area = float(momentos["m00"])

    return contorno, area, cx, cy


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
def control_visual(cx, area):
    global _prev_vel, _integral_x, _prev_err_x

    error_x = (cx - CX_REF) / CX_REF
    _integral_x += error_x
    _integral_x = np.clip(_integral_x, -5.0, 5.0)
    derivativa_x = error_x - _prev_err_x
    _prev_err_x = error_x

    giro = KP_GIRO * error_x + KI_GIRO * _integral_x + KD_GIRO * derivativa_x

    t = np.clip((area - AREA_LEJOS) / (AREA_CERCA - AREA_LEJOS), 0.0, 1.0)
    vel_fwd = VEL_LEJOS + t * (VEL_BASE - VEL_LEJOS)

    factor_centrado = max(0.55, 1.0 - 0.35 * abs(error_x))
    avance = vel_fwd * factor_centrado

    raw_izq = avance + giro
    raw_der = avance - giro
    raw_izq = np.clip(raw_izq, -VEL_MAX, VEL_MAX)
    raw_der = np.clip(raw_der, -VEL_MAX, VEL_MAX)

    left_speed = _prev_vel[0] + ALPHA_VEL * (raw_izq - _prev_vel[0])
    right_speed = _prev_vel[1] + ALPHA_VEL * (raw_der - _prev_vel[1])
    _prev_vel = (left_speed, right_speed)

    return float(left_speed), float(right_speed), float(error_x)


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


def detectar_obstaculo(obs):
    if obs["front"] < OBS_FRONT_NEAR:
        return True
    if obs["side_left"] < OBS_SIDE_NEAR:
        return True
    if obs["side_right"] < OBS_SIDE_NEAR:
        return True
    return False


# ======================================================================
# EVITACIÓN DE OBSTÁCULOS
# ======================================================================
def iniciar_evitacion(obs):
    global _avoid_dir, _avoid_phase, _avoid_steps, _lost_counter, _consecutive_avoids

    _consecutive_avoids += 1

    # Si el robot entra varias veces seguidas en evitación, se asume que está atascado y hay que escapar.
    if _consecutive_avoids >= STUCK_AVOID_LIMIT:
        _avoid_phase = "escape"
        _avoid_steps = ESCAPE_BACK_STEPS
        _lost_counter = SEARCH_MEMORY_STEPS
        reset_control_visual()
        return

    # Elegir lado de evitación una sola vez al entrar.
    if obs["front_right"] <= obs["front_left"] or obs["side_right"] < OBS_SIDE_NEAR:
        _avoid_dir = -1
    else:
        _avoid_dir = 1

    _avoid_phase = "back"
    _avoid_steps = AVOID_BACK_STEPS
    _lost_counter = SEARCH_MEMORY_STEPS
    reset_control_visual()


def control_evitacion(obs):
    global _avoid_phase, _avoid_steps

    if _avoid_phase == "escape":
        if _avoid_steps > 0:
            _avoid_steps -= 1
            return ESCAPE_BACK_SPEED, ESCAPE_BACK_SPEED, "escape_back"

        _avoid_phase = "none"
        return None

    if _avoid_phase == "back":
        if _avoid_steps > 0 and (obs["front"] < OBS_FRONT_NEAR or obs["front"] < OBS_FRONT_HARD):
            _avoid_steps -= 1

            if _avoid_dir == -1:
                return AVOID_BACK_SPEED_STRONG, AVOID_BACK_SPEED_SOFT, "avoid_back_from_right"
            return AVOID_BACK_SPEED_SOFT, AVOID_BACK_SPEED_STRONG, "avoid_back_from_left"

        _avoid_phase = "clear"
        _avoid_steps = AVOID_CLEAR_STEPS

    if _avoid_phase == "clear":
        if _avoid_steps > 0:
            _avoid_steps -= 1

            if _avoid_dir == -1:
                return AVOID_CLEAR_SPEED_FAST, AVOID_CLEAR_SPEED_SLOW, "avoid_clear_right"
            return AVOID_CLEAR_SPEED_SLOW, AVOID_CLEAR_SPEED_FAST, "avoid_clear_left"

        _avoid_phase = "none"
        return None

    return None


# ======================================================================
# BÚSQUEDA
# ======================================================================
def control_busqueda():
    global _lost_counter, _prev_vel

    # Durante la búsqueda con memoria conservamos parte del avance previo.
    if _lost_counter < SEARCH_MEMORY_STEPS:
        _lost_counter += 1

        prev_fwd = 0.5 * (_prev_vel[0] + _prev_vel[1])
        search_fwd = SEARCH_FWD_GAIN * prev_fwd
        search_fwd = saturar(search_fwd, SEARCH_FWD_MIN, SEARCH_FWD_MAX)

        # Se resetea solo el PID para que no arrastre integral/derivada al perder la bola.
        reset_pid_visual_only()

        if _last_seen_side < 0:
            left_speed = search_fwd - SEARCH_TURN_MEMORY
            right_speed = search_fwd + SEARCH_TURN_MEMORY
            return left_speed, right_speed, "search_last_left"

        if _last_seen_side > 0:
            left_speed = search_fwd + SEARCH_TURN_MEMORY
            right_speed = search_fwd - SEARCH_TURN_MEMORY
            return left_speed, right_speed, "search_last_right"

    # Si ya no hay memoria útil, pasa a búsqueda general girando sobre sí mismo.
    reset_control_visual()
    return -SEARCH_TURN_FAST, SEARCH_TURN_FAST, "search_spin"


# ======================================================================
# DEBUG
# ======================================================================
def dibujar_debug(img_rgb, mask, deteccion, obs=None, mode=""):
    debug = img_rgb.copy()
    h, _ = debug.shape[:2]

    cv2.line(debug, (CX_REF, 0), (CX_REF, h - 1), (255, 255, 0), 1)

    if deteccion is not None:
        contorno, area, cx, cy = deteccion
        cv2.drawContours(debug, [contorno], -1, (0, 255, 0), 2)
        cv2.circle(debug, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(debug, f"cx={cx} area={int(area)}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    y0 = 46
    dy = 20

    if mode:
        cv2.putText(debug, f"mode: {mode}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    cv2.putText(debug, f"lost={_lost_counter} last_side={_last_seen_side}", (10, y0 + dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

    if _avoid_phase != "none":
        cv2.putText(debug, f"avoid: {_avoid_phase} dir={_avoid_dir} n={_consecutive_avoids}",
                    (10, y0 + 2 * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 2)

    if obs is not None:
        cv2.putText(debug, f"front={obs['front']:.2f} sideL={obs['side_left']:.2f} sideR={obs['side_right']:.2f}",
                    (10, y0 + 3 * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 2)

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

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)

    coppelia.start_simulation()

    try:
        while coppelia.is_running():
            sonars = robot.get_sonar()
            obs = leer_obstaculos(sonars)

            img = robot.get_image()
            mask = filtro_umbralizacion(img)
            deteccion = detectar_bola(mask)

            if _avoid_phase != "none":
                # Solo tras terminar el primer arco: si reaparece la bola en clear,
                # se abandona evitación y se vuelve a persecución.
                if _avoid_phase == "clear" and deteccion is not None:
                    _avoid_phase = "none"
                    _avoid_steps = 0
                    _, area, cx, _ = deteccion
                    actualizar_memoria_bola(cx)
                    left_speed, right_speed, _ = control_visual(cx, area)
                    mode = "track_after_avoid"
                else:
                    evit = control_evitacion(obs)
                    if evit is not None:
                        left_speed, right_speed, mode = evit
                    else:
                        left_speed, right_speed, mode = control_busqueda()

            elif detectar_obstaculo(obs):
                iniciar_evitacion(obs)
                evit = control_evitacion(obs)
                if evit is not None:
                    left_speed, right_speed, mode = evit
                else:
                    left_speed, right_speed, mode = control_busqueda()

            elif deteccion is not None:
                _, area, cx, _ = deteccion
                actualizar_memoria_bola(cx)
                left_speed, right_speed, _ = control_visual(cx, area)
                mode = "track"

            else:
                left_speed, right_speed, mode = control_busqueda()

            left_speed = saturar(left_speed, -VEL_MAX, VEL_MAX)
            right_speed = saturar(right_speed, -VEL_MAX, VEL_MAX)
            _last_mode = mode

            robot.set_speed(left_speed, right_speed)

            debug = dibujar_debug(img, mask, deteccion, obs, mode)
            cv2.imshow('mascara', mask)
            cv2.imshow('debug', cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        robot.set_speed(0.0, 0.0)
        coppelia.stop_simulation()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()