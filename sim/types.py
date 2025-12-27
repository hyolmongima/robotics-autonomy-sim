from dataclasses import dataclass 
from typing import List, Tuple 


@dataclass
class Pose2D:
	x: float #m
	y: float #m
	yaw: float #radians

@dataclass
class VelocityCommand2D:
	v: float #m/s
	omega: float #rad/s

@dataclass
class Path2D:
	points: List[Tuple[float, float]] # [(x0, y0), (x1, y1), ...] in World frame