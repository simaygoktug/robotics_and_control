#1. Angle Alignment

#Calculate delta_angle = self.agent_orientation - road_angle
#Normalize this to the range [-π, π] to handle wraparound issues
#Use proportional control to steer toward the road direction

#2. Cross-Track Error Correction

#If xte > 0: car is right of center → steer left (negative steering)
#If xte < 0: car is left of center → steer right (positive steering)
#Scale the xte appropriately (dividing by 200) to get reasonable steering values

#3. Combined Control

#Sum both the angle correction and position correction
#Clamp the result to the valid steering range [-1, 1]

#4. Speed Management

#Reduce speed when making sharp turns (large steering angles)
#This helps maintain stability and prevents the car from sliding off the track

#####################################################################################################

import numpy as np
import math,random,sys
from vector import Vector2D

class AutonomousSteeringAgent(SteeringAgent):
    def calculate_driving_decision(self,xte, road_angle, distance_around_track):
        # This function must return a pair of numbers, (target_speed, steering_wheel_position), with 0<=target_speed<=1 and -1<=steering_wheel_position<=1
        # Note that road_angle and self.agent_orientation are in radians.  By comparing them, and allowing for the fact that agent_orientation could have any multiple of 2pi added to it, we can make the steering decision. However we must also consider the xte (cross-track error).
        # If abs(xte)>=track_layout.track_width then it means we've crashed off the road.
        # If xte>0 then we should probably turn left  (depending on the orientation that the car is already in)
        # If xte<0 then we should probably turn right (depending on the orientation that the car is already in)
        
        # Calculate the difference between car orientation and road angle
        delta_angle = self.agent_orientation - road_angle
        
        # Normalize the angle difference to be between -pi and +pi
        delta_angle = delta_angle - ((delta_angle + math.pi) // (2 * math.pi)) * (2 * math.pi)
        
        # Start with basic steering based on angle alignment
        steering_wheel_position = -delta_angle * 0.5  # Proportional control for angle correction
        
        # Add correction based on cross-track error
        # If xte > 0, we're too far right, so steer left (negative steering)
        # If xte < 0, we're too far left, so steer right (positive steering)
        xte_correction = -xte / 200.0  # Scale the xte to reasonable steering values
        
        # Combine angle and position corrections
        steering_wheel_position += xte_correction
        
        # Clamp steering to valid range [-1, 1]
        steering_wheel_position = max(-1.0, min(1.0, steering_wheel_position))
        
        # Adjust speed based on steering magnitude - slow down for sharp turns
        if abs(steering_wheel_position) > 0.5:
            target_speed = 0.6  # Slow down for sharp turns
        elif abs(steering_wheel_position) > 0.3:
            target_speed = 0.8  # Moderate speed for medium turns
        else:
            target_speed = 1.0  # Full speed for straight sections
        
        return (target_speed, steering_wheel_position)