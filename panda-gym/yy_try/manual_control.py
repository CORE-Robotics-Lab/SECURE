import cv2
import copy
def control_robo_keyboard(observation):
    # print(">> Input an action\n")

    while True:
        key = input(">> Input an action\n")
        current_position = observation[0:3]
        DESIRED_MOVE_DIST = 0.1

        if key == "w":
            print(">> up")
            desired_position = copy.deepcopy(current_position)
            desired_position[2] += DESIRED_MOVE_DIST
            action_ratio = control_vel()
            action = action_ratio * (desired_position - current_position)
            return action
        elif key == "s":
            print(">> down")
            desired_position = copy.deepcopy(current_position)
            desired_position[2] -= DESIRED_MOVE_DIST
            action_ratio = control_vel()
            action = action_ratio * (desired_position - current_position)
            return action
        elif key == "a":
            print(">> back")
            desired_position = copy.deepcopy(current_position)
            desired_position[0] -= DESIRED_MOVE_DIST
            action_ratio = control_vel()
            action = action_ratio * (desired_position - current_position)
            return action
        elif key == "d":
            print(">> forward")
            desired_position = copy.deepcopy(current_position)
            desired_position[0] += DESIRED_MOVE_DIST
            action_ratio = control_vel()
            action = action_ratio * (desired_position - current_position)
            return action
        elif key == "q":
            print(">> left")
            desired_position = copy.deepcopy(current_position)
            desired_position[1] += DESIRED_MOVE_DIST
            action_ratio = control_vel()
            action = action_ratio * (desired_position - current_position)
            return action
        elif key == "e":
            print(">> right")
            desired_position = copy.deepcopy(current_position)
            desired_position[1] -= DESIRED_MOVE_DIST
            action_ratio = control_vel()
            action = action_ratio * (desired_position - current_position)
            return action
        elif key == "o":
            print(">> stay")
            desired_position = copy.deepcopy(current_position)
            action = desired_position - current_position
            return action
        else:
            print(">> Other key pressed!")
            continue
        # # 255 is what the console returns when there is no key press...
        # elif key != 255:
        #     print(">> Other key pressed: ", key, ", USELESS!")


def control_vel():
    while True:
        action_ratio = input("Input an action ratio (1-10): \n")
        if action_ratio in [str(x) for x in list(range(1, 11))]:
            print(">> Press ratio ", action_ratio)
            return int(action_ratio)
        else:
            print(">> Other key pressed!")
            continue