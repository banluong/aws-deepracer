import math

def reward_function(params):
    '''
    Trained on Re:Invent2018 track
    Used default Hyperparameters
    Utilizes progress and speed of car, waypoints. Rewards on completion of track.
    Penalizes when car is off track or too much steering.
    '''

    # Read input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    abs_steering = abs(params['steering_angle']) # Only need the absolute steering angle
    steps = params['steps']
    progress = params['progress']
    speed = params['speed']

    # Params for waypoints
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']

    # Total num of steps we want the car to finish the lap, it will vary depends on the track length
    TOTAL_NUM_STEPS = 300
    
    # Initialize reward
    reward = 1.0

    # Penalize if car goes off track
    if not all_wheels_on_track:
        return 1e-3
    
    # Reward if progress is 100%
    if progress == 100:
        reward += 100 
    
    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 15 
    
    # Penalize reward if the car is steering too much
    if abs_steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8
    
    # Give additional reward if the car pass every 100 steps faster than expected
    if (steps % 100) == 0 and progress > (steps / TOTAL_NUM_STEPS) * 100 :
        reward += 10.0
    
    # Reward for progress on track and speed
    if all_wheels_on_track and steps > 0:
        reward += (progress/steps * 100) + speed**2
    else:
        return 1e-3
    
    # ---Waypoints---
    # Calculate the direction of the center line based on the closest waypoints
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # Penalize the reward if the difference is too large
    malus = 1
    DIRECTION_THRESHOLD = 10.0
    if direction_diff > DIRECTION_THRESHOLD:
        malus=1-(direction_diff/50)
        if malus < 0 or malus>1:
            malus = 0
        reward *= malus

    return float(reward)