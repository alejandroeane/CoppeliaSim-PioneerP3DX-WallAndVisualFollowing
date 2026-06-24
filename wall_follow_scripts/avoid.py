import robotica


def avoid(readings):
    izquierda_1 = (readings[0] < 0.2 or readings[1] < 0.4) 
    izquierda_2 = (readings[15] < 0.2 or readings[14] < 0.4) 
    frontal = (readings[3] < 0.4 and readings[4] < 0.4)
    lateral_iz_1 = readings[2] < 0.3
    print(f'izquierda_1: {izquierda_1}, izquierda_2: {izquierda_2}, frontal: {frontal}, lateral_iz_1: {lateral_iz_1}')
    if (izquierda_1 and frontal) or lateral_iz_1:
        lspeed, rspeed = 0.8, -0.2 
    elif not izquierda_1 and izquierda_2 and not frontal:
        lspeed, rspeed = -0.2, +0.8
    else:
        lspeed, rspeed = 0.5, 0.5
    return lspeed, rspeed

def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')
    
    print("Iniciando seguidor de pared con PID...")
    coppelia.start_simulation()
    
    try:
        while coppelia.is_running():
            readings = robot.get_sonar()
            if readings:
                lspeed, rspeed = avoid(readings)
                robot.set_speed(lspeed, rspeed)
    except KeyboardInterrupt:
        pass
    finally:
        coppelia.stop_simulation()

if __name__ == '__main__':
    main()