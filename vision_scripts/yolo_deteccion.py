import cv2
import numpy as np
import robotica
from ultralytics import YOLO

model = YOLO("runs\\detect\\train7\\weights\\best.pt")  
CENTRO = (128, 128)


def filtro_umbralizacion(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bajo1 = np.array([0, 40, 40])   
    alto1 = np.array([10, 255, 255])

    bajo2 = np.array([160, 40, 40])
    alto2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, bajo1, alto1)
    mask2 = cv2.inRange(hsv, bajo2, alto2)
    mask_final = cv2.bitwise_or(mask1, mask2)

    resultado = cv2.bitwise_and(img, img, mask=mask_final)
    return resultado


def detect_yolo_draw(img, model):
    results = model(img, device="cpu",verbose=False, conf=0.5) 
    circulo = None 
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cls = int(box.cls[0])
                # nombre_clase = model.names[cls]
                # cv2.putText(img, nombre_clase, (x1, y1 - 10), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (2, 225, 155), 2)
                cx, cy = (x2+x1)//2, (y2+y1)//2
                cv2.circle(img, (cx,cy), 1, (0,255,0))
                circulo = (cx, cy)
    return circulo 


def giro_derecha(robot, giro):
    robot.set_speed(giro, -giro)


def giro_izquierda(robot, giro):
    robot.set_speed(-giro, giro)


def continuar(robot):
    robot.set_speed(0,0)


def err(cir):
    if cir:
        return cir[0]-CENTRO[0]
    return 0


def seguimiento(robot, circulo):
    error = err(circulo)
    print(error)
    if error >= 40:
        giro_derecha(robot, 0.5)
    elif error <= -40:
        giro_izquierda(robot, 0.5)
    else:
        continuar(robot)
        



def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()
    while coppelia.is_running():
        img = robot.get_image()
        resultado = filtro_umbralizacion(img)
        circulo = detect_yolo_draw(resultado, model=model)
        seguimiento(robot, circulo)
        cv2.imshow('Imagen', resultado)
        # cv2.imshow('opencv', img)
        cv2.waitKey(1)
        # robot.set_speed(0,0)
    coppelia.stop_simulation()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
