import cv2
import numpy as np
import robotica


# ======================================================================
#  PARÁMETROS DEL CONTROLADOR
# ======================================================================
CX_REF = 128             # Centro deseado de la imagen
AREA_MIN = 120.0         # Filtra ruido pequeño
SEARCH_SPEED = 0.8

KP_GIRO    = 0.9         # ganancia proporcional angular
KI_GIRO    = 0.08        # ganancia integral (reducida para evitar windup)
KD_GIRO    = 0.8         # ganancia derivativa

# Umbrales de área ajustados a valores REALES observados (150–3500)
AREA_CERCA = 60000.0      # area >= esto → pelota cerca → VEL_BASE
AREA_LEJOS = 5000.0       # area <= esto → pelota muy lejos → VEL_LEJOS

VEL_BASE   = 1.0         # velocidad de avance cuando la pelota está CERCA
VEL_LEJOS  = 6.0         # velocidad de avance cuando la pelota está LEJOS
VEL_MAX    = 8.0          # velocidad máxima de cada rueda

# Suavizado exponencial (0..1, más alto = respuesta más rápida)
ALPHA_VEL  = 0.4

# ======================================================================
#  ESTADO PERSISTENTE DEL CONTROLADOR
# ======================================================================
_prev_vel   = (0.0, 0.0)         # velocidades anteriores (para suavizado)
_integral_x = 0.0                # acumulador integral del error lateral
_prev_err_x = 0.0                # error lateral anterior (para derivativa)


def saturar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def filtro_umbralizacion(img_rgb):
    """
    Segmentación en HSV de una bola roja. La imagen recibida desde robotica.py está en RGB.
    """
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
    """
    Devuelve 'contorno', 'area', 'cx' y 'cy' si encuentra la bola. None en caso contrario.
    """
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

    factor_lat = 1.0 - 0.5 * min(abs(t), 1.0)
    avance = vel_fwd * factor_lat

    raw_izq = avance + giro
    raw_der = avance - giro
    raw_izq = np.clip(raw_izq, -VEL_MAX, VEL_MAX)
    raw_der = np.clip(raw_der, -VEL_MAX, VEL_MAX)

    left_speed = _prev_vel[0] + ALPHA_VEL * (raw_izq - _prev_vel[0])
    right_speed = _prev_vel[1] + ALPHA_VEL * (raw_der - _prev_vel[1])
    _prev_vel = (left_speed, right_speed)

    return float(left_speed), float(right_speed), error_x


def dibujar_debug(img_rgb, mask, deteccion):
    debug = img_rgb.copy()
    h, w = debug.shape[:2]

    cv2.line(debug, (CX_REF, 0), (CX_REF, h - 1), (255, 255, 0), 1)

    if deteccion is not None:
        contorno, area, cx, cy = deteccion
        cv2.drawContours(debug, [contorno], -1, (0, 255, 0), 2)
        cv2.circle(debug, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(
            debug,
            f"area={int(area)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return debug


def main(args=None):
    global _prev_vel, _integral_x, _prev_err_x

    # Reiniciar estado por si se reutiliza el módulo
    _prev_vel = (0.0, 0.0)
    _integral_x = 0.0
    _prev_err_x = 0.0

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)

    coppelia.start_simulation()

    try:
        while coppelia.is_running():
            img = robot.get_image()
            mask = filtro_umbralizacion(img)
            deteccion = detectar_bola(mask)

            if deteccion is None:
                # Si no veo la bola, giro para buscarla
                left_speed = -SEARCH_SPEED
                right_speed = SEARCH_SPEED
            else:
                _, area, cx, _ = deteccion
                left_speed, right_speed, _ = control_visual(cx, area)

            robot.set_speed(left_speed, right_speed)

            debug = dibujar_debug(img, mask, deteccion)

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