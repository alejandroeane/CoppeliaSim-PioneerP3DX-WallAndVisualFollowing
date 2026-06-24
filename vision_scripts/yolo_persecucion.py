from yolo_deteccion import err
import cv2
import numpy as np
import robotica
from ultralytics import YOLO

model = YOLO("runs\\detect\\train7\\weights\\best.pt")

# --- Resolución de cámara ---
IMG_W = 256
IMG_H = 256
CENTRO_Y = IMG_H // 2          # 128
CENTRO_X = IMG_W // 2          # 128

# ======================================================================
#  PARÁMETROS DEL CONTROLADOR  –  ajustar aquí
# ======================================================================
KP_GIRO    = 0.9       # ganancia proporcional angular (rad/s por unidad de error normalizado)
KI_GIRO    = 0.4       # ganancia integral (corrige error estacionario de giro)
KD_GIRO    = 1.4      # ganancia derivativa (amortigua oscilaciones de giro)

VEL_BASE   = 4       # velocidad de avance cuando la pelota está CERCA
VEL_LEJOS  = 5       # velocidad de avance cuando la pelota está LEJOS
VEL_MAX    = 6       # velocidad máxima de cada rueda

# Umbrales de cercanía basados en la altura del bbox / IMG_H
H_RATIO_CERCA = 0.4   # h_ratio >= esto → pelota cerca → VEL_BASE
H_RATIO_LEJOS = 0.1   # h_ratio <= esto → pelota muy lejos → VEL_LEJOS

# Suavizado exponencial para las velocidades (0..1, más bajo = más suave)
ALPHA_VEL  = 0.4

# ======================================================================
#  ESTADO PERSISTENTE DEL CONTROLADOR
# ======================================================================
_prev_vel   = (0.0, 0.0)         # velocidades anteriores (para suavizado)
_integral_x = 0.0                # acumulador integral del error lateral
_prev_err_x = 0.0                # error lateral anterior (para derivativa)
_perdida_counter = 0             # frames sin detección
_ultima_dir = 1                  # dirección de búsqueda (+1 = derecha, -1 = izquierd/a)


# ======================================================================
#  FILTRO DE COLOR (previa a YOLO para reducir falsos positivos)
# ======================================================================
def filtro_umbralizacion(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bajo1 = np.array([0,   40,  40])
    alto1 = np.array([10,  255, 255])
    bajo2 = np.array([160, 40,  40])
    alto2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, bajo1, alto1)
    mask2 = cv2.inRange(hsv, bajo2, alto2)
    mask_final = cv2.bitwise_or(mask1, mask2)

    resultado = cv2.bitwise_and(img, img, mask=mask_final)
    return resultado


# ======================================================================
#  DETECCIÓN YOLO
# ======================================================================
def detect_yolo_draw(img, model):
    """Devuelve (cx, cy, h_bbox) del bbox con mayor confianza, o None.
    h_bbox = y2 - y1 (altura del bounding box en píxeles).
    """
    results = model(img, device="cpu", verbose=False, conf=0.4)
    mejor = None
    mejor_conf = 0.0

    for result in results:
        if result.boxes:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > mejor_conf:
                    mejor_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx  = (x1 + x2) // 2
                    cy  = (y1 + y2) // 2
                    h_bbox = y2 - y1
                    mejor = (cx, cy, h_bbox, x1, y1, x2, y2)

    if mejor:
        cx, cy, h_bbox, x1, y1, x2, y2 = mejor
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
        cv2.circle(img, (CENTRO_X, IMG_H // 2), 4, (0, 0, 255), -1)
        cv2.line(img, (CENTRO_X, IMG_H // 2), (cx, cy), (255, 255, 0), 1)
        return (cx, cy, h_bbox, y2-y1)

    return None


# ======================================================================
#  CONTROLADOR PID + AVANCE PROPORCIONAL
# ======================================================================
def calcular_velocidades(deteccion):
    global _prev_vel, _integral_x, _prev_err_x, _perdida_counter, _ultima_dir

    if deteccion is None:
        return -1,1

    _perdida_counter = 0
    cx, cy, h_bbox, h_pixels = deteccion

    error_y = h_pixels / CENTRO_Y     
    error_x = (cx - CENTRO_X) / CENTRO_X
    if abs(error_y) > 0.1:
        _ultima_dir = 1 if error_y > 0 else -1

    _integral_x += error_x
    _integral_x = np.clip(_integral_x, -10.0, 10.0)        
    derivativa_x = error_x - _prev_err_x
    _prev_err_x = error_x

    giro = KP_GIRO * error_x + KI_GIRO * _integral_x + KD_GIRO * derivativa_x

    h_ratio = h_bbox / IMG_H   
    t = np.clip((h_ratio - H_RATIO_LEJOS) / (H_RATIO_CERCA - H_RATIO_LEJOS),
                0.0, 1.0)
    vel_fwd = VEL_LEJOS + t * (VEL_BASE - VEL_LEJOS)

    factor_lat = 1.0 - 0.3 * min(abs(error_y), 1.0)
    vel_fwd *= factor_lat

    raw_izq = vel_fwd + giro
    raw_der = vel_fwd - giro

    raw_izq = np.clip(raw_izq, -VEL_MAX, VEL_MAX)
    raw_der = np.clip(raw_der, -VEL_MAX, VEL_MAX)

    vel_izq = _prev_vel[0] + ALPHA_VEL * (raw_izq - _prev_vel[0])
    vel_der = _prev_vel[1] + ALPHA_VEL * (raw_der - _prev_vel[1])
    _prev_vel = (vel_izq, vel_der)

    return float(vel_izq), float(vel_der)


# ======================================================================
#  BUCLE PRINCIPAL
# ======================================================================
def main(args=None):
    global _prev_vel, _integral_x, _prev_err_x, _perdida_counter, _ultima_dir

    # Reiniciar estado por si se reutiliza el módulo
    _prev_vel = (0.0, 0.0)
    _integral_x = 0.0
    _prev_err_x = 0.0
    _perdida_counter = 0
    _ultima_dir = 1

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()

    while coppelia.is_running():
        img = robot.get_image()
        img_filtrada = filtro_umbralizacion(img.copy())
        deteccion    = detect_yolo_draw(img_filtrada, model=model)

        vel_izq, vel_der = calcular_velocidades(deteccion)
        robot.set_speed(vel_izq, vel_der)

        # HUD de depuración
        if deteccion:
            cx, cy, h_bbox, h_pixels = deteccion
            error_x = (cx - CENTRO_X) / CENTRO_X
            h_ratio = h_bbox / IMG_H
            estado = (f"err={error_x:+.2f}  h={h_ratio:.2f}  "
                      f"L={vel_izq:.2f}  R={vel_der:.2f}")
        else:
            estado = f"BUSCANDO  L={vel_izq:.2f}  R={vel_der:.2f}"

        cv2.putText(img_filtrada, estado, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        cv2.imshow('Seguimiento YOLO', img_filtrada)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    coppelia.stop_simulation()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()