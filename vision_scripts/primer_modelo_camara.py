import cv2
import numpy as np
import robotica


# Parámetros del controlador
CX_REF = 128             # Centro deseado de la imagen
AREA_MIN = 120.0         # Filtra ruido pequeño
K_GIRO = 0.012           # Ganancia para centrar la bola
K_AVANCE = 0.00035       # Ganancia para acercarse / alejarse según el área
SEARCH_SPEED = 0.8       # Giro de búsqueda cuando no se detecta bola
MAX_SPEED = 5.0          # Saturación de velocidad de rueda


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
    error_x = CX_REF - cx

    # Giro proporcional más contenido
    giro = 0.008 * error_x

    # Avance fuerte si la bola está lejos
    if area < 1200:
        avance = 6.5
    elif area < 2500:
        avance = 5.8
    elif area < 4500:
        avance = 5.1
    else:
        avance = 4.4

   # Reducir avance si está descentrada
    if abs(error_x) > 80:
        avance *= 0.65
    elif abs(error_x) > 40:
        avance *= 0.85

    left_speed = saturar(avance - giro, -MAX_SPEED, MAX_SPEED)
    right_speed = saturar(avance + giro, -MAX_SPEED, MAX_SPEED)

    return left_speed, right_speed, error_x


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
            f"cx={cx} cy={cy} area={int(area)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return debug


def main(args=None):
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