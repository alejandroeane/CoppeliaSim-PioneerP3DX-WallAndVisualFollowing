import time
from enum import Enum, auto

import robotica


# =========================
# CONFIGURACIÓN GENERAL
# =========================
FOLLOW_LEFT = True

DESIRED_WALL_DISTANCE = 0.25
WALL_DETECT_DISTANCE = 0.35
WALL_LOST_DISTANCE = 0.70

FRONT_STOP_DISTANCE = 0.3
FRONT_CLEAR_DISTANCE = 0.42

BASE_SPEED = 1.8
SEARCH_SPEED = 1.0
TURN_SPEED = 1.2
MAX_SPEED = 3

ANGLE_WEIGHT = 1.2
# KP = 0.4857
# KI = 0.2270
# KD = 0.9283
KP = 2.5
KI = 0.01
KD = 0.8


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
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error, dt):
        if dt <= 1e-6:
            return 0.0

        if not self.initialized:
            self.prev_error = error
            self.initialized = True

        self.integral += error * dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(u, self.out_min, self.out_max)


# =========================
# UTILS
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def saturate_speeds(left, right):
    return clamp(left, -MAX_SPEED, MAX_SPEED), clamp(right, -MAX_SPEED, MAX_SPEED)


def sonar_groups(readings, follow_left=True):
    front = min(readings[3], readings[4])

    if follow_left:
        side_front = min(readings[0], readings[1], readings[2])
        side_rear = min(readings[15], readings[14])
    else:
        side_front = min(readings[5], readings[6], readings[7])
        side_rear = min(readings[10], readings[11])

    side_mean = 0.5 * (side_front + side_rear)
    return front, side_front, side_rear, side_mean


def wall_follow_step(readings, pid, state, dt, follow_left=True):
    front, side_front, side_rear, side_mean = sonar_groups(readings, follow_left)

    wall_seen = side_mean < WALL_DETECT_DISTANCE
    wall_lost = side_front > WALL_LOST_DISTANCE and side_rear > WALL_LOST_DISTANCE

    left_speed = 0.0
    right_speed = 0.0
    new_state = state

    if state == State.SEARCH_WALL:
        pid.reset()

        if front < FRONT_STOP_DISTANCE:
            new_state = State.AVOID_FRONT
        elif wall_seen:
            new_state = State.FOLLOW_WALL
        else:
            if follow_left:
                left_speed, right_speed = SEARCH_SPEED * 0.55, SEARCH_SPEED
            else:
                left_speed, right_speed = SEARCH_SPEED, SEARCH_SPEED * 0.55

    elif state == State.FOLLOW_WALL:
        if front < FRONT_STOP_DISTANCE:
            pid.reset()
            new_state = State.AVOID_FRONT
        elif wall_lost:
            pid.reset()
            new_state = State.TURN_TO_WALL
        else:
            error_dist = DESIRED_WALL_DISTANCE - side_mean

            if follow_left:
                error_angle = side_rear - side_front
            else:
                error_angle = side_front - side_rear

            error = error_dist + ANGLE_WEIGHT * error_angle
            correction = pid.update(error, dt)

            slowdown = 1.0
            if front < FRONT_CLEAR_DISTANCE:
                ratio = (front - FRONT_STOP_DISTANCE) / (FRONT_CLEAR_DISTANCE - FRONT_STOP_DISTANCE)
                slowdown = clamp(ratio, 0.45, 1.0)

            if follow_left:
                left_speed = (BASE_SPEED + correction) * slowdown
                right_speed = (BASE_SPEED - correction) * slowdown
            else:
                left_speed = (BASE_SPEED - correction) * slowdown
                right_speed = (BASE_SPEED + correction) * slowdown

    elif state == State.TURN_TO_WALL:
        pid.reset()

        if front < FRONT_STOP_DISTANCE:
            new_state = State.AVOID_FRONT
        elif wall_seen:
            new_state = State.FOLLOW_WALL
        else:
            if follow_left:
                left_speed, right_speed = 0.35, 1.0
            else:
                left_speed, right_speed = 1.0, 0.35

    elif state == State.AVOID_FRONT:
        pid.reset()

        if front > FRONT_CLEAR_DISTANCE:
            if wall_seen:
                new_state = State.FOLLOW_WALL
            else:
                new_state = State.SEARCH_WALL

        if follow_left:
            left_speed, right_speed = TURN_SPEED, -TURN_SPEED
        else:
            left_speed, right_speed = -TURN_SPEED, TURN_SPEED

    left_speed, right_speed = saturate_speeds(left_speed, right_speed)
    return left_speed, right_speed, new_state, front, side_front, side_rear, side_mean


# =========================
# CONTROL PRINCIPAL
# =========================
def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')

    pid = PID(KP, KI, KD)
    state = State.SEARCH_WALL

    print("Iniciando wall following con PID + FSM...")
    coppelia.start_simulation()

    last_time = time.time()
    cycle = 0

    try:
        while coppelia.is_running():
            readings = robot.get_sonar()

            if readings and len(readings) >= 16:
                now = time.time()
                dt = now - last_time
                last_time = now

                lspeed, rspeed, state, front, side_f, side_r, side_m = wall_follow_step(
                    readings, pid, state, dt, FOLLOW_LEFT
                )

                robot.set_speed(lspeed, rspeed)

                cycle += 1
                if cycle % 20 == 0:
                    print(
                        f"state={state.name:12s} | "
                        f"front={front:.3f} | sideF={side_f:.3f} | "
                        f"sideR={side_r:.3f} | sideM={side_m:.3f} | "
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
