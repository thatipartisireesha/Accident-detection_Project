import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
import datetime
import os

# ---------------- LOAD MODEL ---------------- #

model = YOLO("yolov8n.pt")

vehicle_classes = [2,3,5,7]   # car bike bus truck
person_class = 0

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
MIN_VEHICLE_SIZE = 2000     # Minimum bounding box area to avoid distant/small detections (lowered to avoid missing smaller targets)
IOU_COLLISION_THRESHOLD = 0.40  # stricter overlap required to reduce false positives on close non-collision parked vehicles
DISTANCE_CRASH_RELATIVE_THRESHOLD = 0.9  # even tighter for dangerous proximity
DIVIDER_ZONE_RATIO = 0.35   # Fraction of width used to represent median/divider collision zone (wider to catch divider contact)
DIVIDER_OVERLAP_THRESHOLD = 0.10  # Fraction of bbox area that must overlap with divider zone
DIVIDER_MIN_BBOX_AREA = 10000     # Ignore tiny vehicles near divider to avoid irrelevant overlaps
PARKING_SCENE_CONFIDENCE = 0.6    # proportion of parking-style pairs required to treat as parking environment

# ---------------- CAMERA LOCATION ---------------- #

camera_locations = {
    "Camera 1":"Highway Junction",
    "Camera 2":"City Main Road",
    "Camera 3":"Traffic Signal",
    "Camera 4":"Urban Road"
}

# ---------------- IOU ---------------- #

def calculate_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB-xA) * max(0, yB-yA)

    if interArea == 0:
        return 0

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea/(boxAArea+boxBArea-interArea)

# ---------------- VEHICLE COLLISION ---------------- #

def vehicle_collision(vehicles):
    if is_parking_scene(vehicles):
        return False

    for i in range(len(vehicles)):
        for j in range(i+1,len(vehicles)):

            boxA = vehicles[i]
            boxB = vehicles[j]

            iou = calculate_iou(boxA,boxB)
            if iou <= IOU_COLLISION_THRESHOLD:
                continue

            centerAx = (boxA[0]+boxA[2]) / 2
            centerAy = (boxA[1]+boxA[3]) / 2
            centerBx = (boxB[0]+boxB[2]) / 2
            centerBy = (boxB[1]+boxB[3]) / 2
            pixel_distance = ((centerAx-centerBx)**2 + (centerAy-centerBy)**2)**0.5

            avg_width = ((boxA[2]-boxA[0]) + (boxB[2]-boxB[0])) / 2
            avg_height = ((boxA[3]-boxA[1]) + (boxB[3]-boxB[1])) / 2

            # reject false grouping where IoU is from detection bleed (minor overlap) in long-range scene
            if pixel_distance > avg_width * 1.2 or abs(centerAy-centerBy) > avg_height * 1.0:
                continue

            return True

    return False

# ---------------- VEHICLE DISTANCE CRASH ---------------- #

def vehicle_distance_crash(vehicles, confidences=None):
    """
    Improved distance analysis for toll plaza scenarios.
    Considers relative positioning and vehicle sizes to reduce false positives.
    Uses normalized distances for scale-invariant comparison.
    """
    if len(vehicles) < 2:
        return False

    if is_parking_scene(vehicles):
        return False
    
    if confidences is None:
        confidences = [0.5] * len(vehicles)

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            boxA = vehicles[i]
            boxB = vehicles[j]
            
            conf_a = confidences[i] if i < len(confidences) else 0.5
            conf_b = confidences[j] if j < len(confidences) else 0.5

            # Use normalized distance calculation
            norm_dist, pixel_distance = calculate_normalized_distance(boxA, boxB)
            
            # For crash detection: normalized distance should be < 2.0 (vehicles very close)
            # Higher confidence in detection = stricter distance threshold
            confidence_factor = (conf_a + conf_b) / 2.0
            effective_threshold = DISTANCE_CRASH_RELATIVE_THRESHOLD * confidence_factor
            
            if norm_dist >= effective_threshold:
                continue

            centerAx = (boxA[0] + boxA[2]) / 2
            centerAy = (boxA[1] + boxA[3]) / 2
            centerBx = (boxB[0] + boxB[2]) / 2
            centerBy = (boxB[1] + boxB[3]) / 2

            avg_width = ((boxA[2] - boxA[0]) + (boxB[2] - boxB[0])) / 2
            avg_height = ((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2

            # reject false grouping where IoU is from detection bleed (minor overlap) in long-range scene
            if pixel_distance > avg_width * 1.2 or abs(centerAy - centerBy) > avg_height * 1.0:
                continue

            dy = abs(centerAy - centerBy)
            dx = abs(centerAx - centerBx)

            # generally expect collision candidates to be aligned along driving direction, not side-by-side in adjacent lanes
            if dx > 1.5 * avg_width and dy < 0.25 * avg_height:
                continue

            vertical_overlap = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])
            horizontal_overlap = min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])

            if (vertical_overlap > 0.5 * avg_height) or (horizontal_overlap > 0.5 * avg_width):
                return True

    return False

# ---------------- DIVIDER COLLISION ---------------- #

def divider_collision(vehicles, frame_shape, zone_ratio=DIVIDER_ZONE_RATIO, overlap_threshold=DIVIDER_OVERLAP_THRESHOLD):
    """Detect collisions with a divider/median based on vehicle bounding box overlap.

    This detects vehicles that intersect divider zones (left edge, center median, right edge).
    It compares the intersection area against a fraction of the vehicle bounding box area.
    """
    if len(vehicles) == 0:
        return False

    height, width = frame_shape[0], frame_shape[1]

    left_zone = (0, width * zone_ratio)
    center_zone = (width * (0.5 - zone_ratio / 2), width * (0.5 + zone_ratio / 2))
    right_zone = (width * (1 - zone_ratio), width)

    # Consider vehicles that are closer to the camera (lower-to-middle part of the image)
    vertical_threshold = height * 0.30

    for v in vehicles:
        x1, y1, x2, y2 = v
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        if bbox_area == 0:
            continue

        center_y = (y1 + y2) / 2
        if center_y < vertical_threshold:
            continue

        for zone in (left_zone, center_zone, right_zone):
            overlap_x1 = max(x1, zone[0])
            overlap_x2 = min(x2, zone[1])
            overlap_width = max(0, overlap_x2 - overlap_x1)
            overlap_area = overlap_width * (y2 - y1)

            # Require significant overlap and a minimum vehicle size
            if (overlap_area / bbox_area >= overlap_threshold) and (bbox_area >= DIVIDER_MIN_BBOX_AREA):
                return True

    return False


def adjusted_divider_collision(vehicles, frame_shape):
    """Divider collision with parking and small-object false-positive filtering."""
    if is_parking_scene(vehicles):
        return False

    # Apply same rule set with minimum vehicle size
    filtered_vehicles = [v for v in vehicles if max(0, v[2] - v[0]) * max(0, v[3] - v[1]) >= DIVIDER_MIN_BBOX_AREA]
    return divider_collision(filtered_vehicles, frame_shape)

# ---------------- SINGLE VEHICLE IMPACT ---------------- #

def single_vehicle_impact(vehicles, frame_shape):
    """Detect high-confidence single-vehicle accidents (severe crash/rollover).
    
    Only triggers for DAMAGED vehicles, not just large ones.
    A large parked vehicle is NOT an accident.
    """
    if len(vehicles) != 1:
        return False

    height, width = frame_shape[0], frame_shape[1]
    frame_area = width * height

    x1, y1, x2, y2 = vehicles[0]
    vehicle_area = max(0, x2 - x1) * max(0, y2 - y1)
    w = x2 - x1
    h = y2 - y1 + 1e-9

    # Only flag if vehicle shows ABNORMAL POSTURE (flipped, rolled, crashed)
    # NOT just because it's large
    aspect_ratio = w / h
    
    # Flipped/rolled vehicle: extremely wide and covers significant area
    if vehicle_area >= 0.25 * frame_area and aspect_ratio > 1.5:
        return True

    # Very large + off-center (crashed into barrier): 28%+ of frame AND abnormal aspect
    if vehicle_area >= 0.28 * frame_area and (aspect_ratio > 1.4 or aspect_ratio < 0.6):
        return True
    
    # NEW: Vehicle in corner/edge position with sizeable area (indicates impact/collision)
    # Vehicle pushed to left/right edge
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Severe edge positioning with large vehicle = collision
    if vehicle_area >= 0.18 * frame_area:
        # Vehicle in left corner (pushed by impact)
        if center_x < width * 0.3 and aspect_ratio > 1.2:
            return True
        # Vehicle in right corner (pushed by impact)  
        if center_x > width * 0.7 and aspect_ratio > 1.2:
            return True
        # Vehicle in bottom corner (severe impact)
        if center_y > height * 0.65 and (aspect_ratio > 1.3 or aspect_ratio < 0.75):
            return True
    
    # NEW: Severely collapsed/damaged single vehicle (even if medium-sized)
    # PRIORITY: High severity damage indicators
    if vehicle_area >= 0.12 * frame_area:  # Medium-sized vehicle
        # Severely crushed/collapsed: extreme aspect ratios
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            return True
        # Highly deformed with moderate size
        if vehicle_area >= 0.15 * frame_area and (aspect_ratio > 1.6 or aspect_ratio < 0.6):
            return True
        # Severely deformed medium vehicle
        if vehicle_area >= 0.13 * frame_area and (aspect_ratio > 1.8 or aspect_ratio < 0.55):
            return True

    return False


def heavy_vehicle_impact(vehicles, frame_shape):
    """Fallback for large truck/bus objects showing crash/rollover posture.
    
    Only triggers for actual damage/abnormal positioning, not just size.
    Enhanced to detect severe collisions like bus-equipment impacts.
    """
    if len(vehicles) == 0:
        return False

    height, width = frame_shape[0], frame_shape[1]
    frame_area = width * height

    for v in vehicles:
        x1, y1, x2, y2 = v
        vehicle_area = max(0, x2 - x1) * max(0, y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1 + 1e-9
        aspect_ratio = w / h

        # Very large vehicle (>25% frame) centered with abnormal aspect = crash
        if vehicle_area >= 0.25 * frame_area and abs(center_x - width / 2) < 0.4 * width and abs(center_y - height / 2) < 0.45 * height:
            if aspect_ratio > 1.4 or aspect_ratio < 0.6:  # Abnormal: either very wide or very tall
                return True

        # Wide collapsed vehicle (flipped/rolled): >20% area AND width > 1.5x height
        if vehicle_area >= 0.20 * frame_area and aspect_ratio > 1.5:
            return True
        
        # NEW: Enhanced heavy vehicle impact detection
        # Large vehicle (bus/truck) with severe positioning = collision
        if vehicle_area >= 0.18 * frame_area:
            # Vehicle pushed to edge (side impact)
            if center_x < width * 0.25 or center_x > width * 0.75:
                if aspect_ratio > 1.1 or aspect_ratio < 0.8:
                    return True
            
            # Very large vehicle with significant deformation (extreme aspect ratio)
            if vehicle_area >= 0.22 * frame_area and (aspect_ratio > 1.8 or aspect_ratio < 0.55):
                return True
            
            # Large vehicle in bottom-heavy position (severe impact)
            if vehicle_area >= 0.20 * frame_area and center_y > height * 0.55:
                if aspect_ratio > 1.2 or aspect_ratio < 0.7:
                    return True

    return False

# NEW: Severe positioning accident detection
def detect_severe_positioning_accident(vehicles, frame_shape, all_detections=None):
    """
    Detect accidents where vehicles are in severe/abnormal positions.
    Examples: Vehicle pushed to corner, vehicle in ditch, vehicle at odd angles.
    Handles scenarios like bus-equipment collisions.
    """
    if len(vehicles) == 0:
        return False
    
    height, width = frame_shape[0], frame_shape[1]
    frame_area = width * height
    
    for v in vehicles:
        x1, y1, x2, y2 = v
        vehicle_area = max(0, x2 - x1) * max(0, y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1 + 1e-9
        aspect_ratio = w / h
        
        # Sizeable vehicle in extreme positions = likely collision
        if vehicle_area >= 0.15 * frame_area:
            # Vehicle in left corner (pushed by equipment/barrier on right)
            left_corner = center_x < width * 0.2 and center_y > height * 0.3
            # Vehicle in right corner (pushed by equipment/barrier on left)
            right_corner = center_x > width * 0.8 and center_y > height * 0.3
            # Vehicle at bottom corner (severe impact pushing down)
            bottom_corner = center_y > height * 0.6 and (center_x < width * 0.3 or center_x > width * 0.7)
            # Vehicle very low in frame with abnormal aspect (deformation)
            bottom_deformed = center_y > height * 0.65 and (aspect_ratio > 1.4 or aspect_ratio < 0.65)
            
            if (left_corner or right_corner or bottom_corner or bottom_deformed):
                return True
        
        # Very large vehicle with extreme aspect ratio (severe deformation)
        if vehicle_area >= 0.20 * frame_area and (aspect_ratio > 1.6 or aspect_ratio < 0.6):
            return True
    
    return False


def detect_wreck_cluster(vehicles, frame_shape):
    """
    Detect chaotic accident scenes with multiple vehicles in collision.
    Vehicles clustered together in lower frame area = likely multi-vehicle wreck.
    """
    if len(vehicles) < 2:
        return False
    
    height, width = frame_shape[0], frame_shape[1]
    frame_area = width * height
    
    # Analyze clustering in lower frame (where accidents typically occur)
    lower_threshold = height * 0.4
    vehicles_in_lower = []
    for v in vehicles:
        center_y = (v[1] + v[3]) / 2
        if center_y > lower_threshold:
            v_area = max(0, v[2] - v[0]) * max(0, v[3] - v[1])
            if v_area >= 1500:  # Medium-sized vehicles
                vehicles_in_lower.append(v)
    
    # If multiple sizeable vehicles concentrated in lower area = wreck
    if len(vehicles_in_lower) >= 2:
        # Calculate clustering tightness
        x_coords = []
        y_coords = []
        for v in vehicles_in_lower:
            center_x = (v[0] + v[2]) / 2
            center_y = (v[1] + v[3]) / 2
            x_coords.append(center_x)
            y_coords.append(center_y)
        
        x_span = max(x_coords) - min(x_coords) if x_coords else width
        y_span = max(y_coords) - min(y_coords) if y_coords else height
        
        # Tightly clustered vehicles (small span) = collision
        if x_span < width * 0.6 and y_span < height * 0.35:
            return True
        
        # Very close together vertically = multi-vehicle collision
        if y_span < height * 0.2 and len(vehicles_in_lower) >= 2:
            return True
    
    return False

# ---------------- COLLAPSED WRECK SCENE ---------------- #

def collapsed_wreck_fallback(vehicles, persons, frame_shape):
    """Final fallback for large static wreckage environments.
    
    Enhanced to detect multi-vehicle wrecks and bus-equipment collisions.
    Does NOT trigger just because persons are present - only for actual wreckage.
    """
    height, width = frame_shape[0], frame_shape[1]
    frame_area = height * width

    if len(vehicles) == 0:
        return False

    # Check for wreck clusters (multiple vehicles in chaotic collision)
    if detect_wreck_cluster(vehicles, frame_shape):
        return True

    for v in vehicles:
        x1, y1, x2, y2 = v
        v_area = max(0, x2 - x1) * max(0, y2 - y1)
        w = x2 - x1
        h = y2 - y1 + 1e-9
        aspect_ratio = w / h
        
        # Only large vehicles taking up significant frame area
        if v_area >= 0.22 * frame_area:
            return True

        # collapsed/wide wreck posture (flipped or rolled vehicle) 
        if v_area >= 0.15 * frame_area and aspect_ratio > 1.3:
            return True
        
        # NEW: Severe deformation of medium-sized vehicle (bus-equipment collision)
        if v_area >= 0.12 * frame_area and (aspect_ratio > 1.5 or aspect_ratio < 0.65):
            return True

    return False

# ---------------- HUMAN NEAR VEHICLE ---------------- #
def is_parking_style_pair(boxA, boxB):
    """Heuristic check to identify parked vehicles side-by-side in a static parking layout."""
    # Center positions
    centerAx = (boxA[0] + boxA[2]) / 2
    centerAy = (boxA[1] + boxA[3]) / 2
    centerBx = (boxB[0] + boxB[2]) / 2
    centerBy = (boxB[1] + boxB[3]) / 2

    # Size metrics
    heightA = boxA[3] - boxA[1]
    heightB = boxB[3] - boxB[1]
    widthA = boxA[2] - boxA[0]
    widthB = boxB[2] - boxB[0]

    avg_height = (heightA + heightB) / 2
    avg_width = (widthA + widthB) / 2

    dy = abs(centerAy - centerBy)
    dx = abs(centerAx - centerBx)
    
    # Check if vehicles are roughly same size (parked vehicles)
    size_ratio = max(heightA, heightB) / (min(heightA, heightB) + 1e-9)
    if size_ratio > 1.5:
        return False

    # Side-by-side parked: separated horizontally with minimal vertical difference
    if avg_height > 0 and dy < 0.2 * avg_height and dx > 0.3 * avg_width and dx < 2.0 * avg_width:
        vertical_overlap = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])
        if vertical_overlap > 0.7 * max(heightA, heightB):
            return True
    
    # Front-to-back parked
    if avg_width > 0 and dx < 0.3 * avg_width and dy > 0.3 * avg_height and dy < 2.0 * avg_height:
        horizontal_overlap = min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])
        if horizontal_overlap > 0.6 * max(widthA, widthB):
            return True

    return False


def is_parking_scene(vehicles):
    """Determines if the detected cars form a parked-vehicles layout to reduce false positives."""
    if len(vehicles) < 2:
        return False
    
    # For scenes with many vehicles (>=4), more likely to be parking lot
    if len(vehicles) >= 4:
        park_like_pairs = 0
        total_pairs = 0
        
        for i in range(len(vehicles)):
            for j in range(i+1, len(vehicles)):
                total_pairs += 1
                if is_parking_style_pair(vehicles[i], vehicles[j]):
                    park_like_pairs += 1
        
        # For parking lots with 4+ vehicles: require only 40% of pairs
        parking_threshold = max(1, total_pairs * 0.4)
        return park_like_pairs >= parking_threshold
    
    # For 2-3 vehicles, use stricter parking pair matching
    park_like_pairs = 0
    for i in range(len(vehicles)):
        for j in range(i+1, len(vehicles)):
            if is_parking_style_pair(vehicles[i], vehicles[j]):
                park_like_pairs += 1
    
    # At least one parking-style pair for small scenes
    return park_like_pairs >= 1


def calculate_normalized_distance(box1, box2):
    """
    Calculate normalized distance between two objects considering their sizes.
    Returns distance normalized by average object size for scale-invariant comparison.
    """
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    
    pixel_distance = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
    
    # Calculate average size
    size1 = max(box1[2] - box1[0], box1[3] - box1[1])
    size2 = max(box2[2] - box2[0], box2[3] - box2[1])
    avg_size = (size1 + size2) / 2.0
    
    if avg_size < 1:
        return float('inf')
    
    normalized_distance = pixel_distance / avg_size
    return normalized_distance, pixel_distance


def human_near_vehicle(vehicles, persons, confidences_vehicles=None, confidences_persons=None):
    """
    Strict detection of humans at IMMEDIATE risk near vehicles.
    Only triggers for actual emergency scenarios (people touching/inside vehicles).
    This avoids false positives on street scenes with pedestrians.
    """
    if len(vehicles) == 0 or len(persons) == 0:
        return False
    
    # Normalize confidence arrays if provided
    if confidences_vehicles is None:
        confidences_vehicles = [0.5] * len(vehicles)
    if confidences_persons is None:
        confidences_persons = [0.5] * len(persons)
    
    # Only flag if MULTIPLE persons in VERY CLOSE proximity to vehicles
    very_close_person_vehicle_pairs = 0
    
    for i, v in enumerate(vehicles):
        v_conf = confidences_vehicles[i] if i < len(confidences_vehicles) else 0.5
        
        # Only consider high-confidence vehicle detections
        if v_conf < 0.7:  # Higher threshold for vehicle
            continue
        
        vx = (v[0] + v[2]) / 2
        vy = (v[1] + v[3]) / 2
        v_width = v[2] - v[0]
        v_height = v[3] - v[1]
        
        for j, p in enumerate(persons):
            p_conf = confidences_persons[j] if j < len(confidences_persons) else 0.5
            
            # Only consider VERY high-confidence person detections
            if p_conf < 0.7:  # Higher threshold for person
                continue
            
            px = (p[0] + p[2]) / 2
            py = (p[1] + p[3]) / 2
            
            # Calculate actual pixel distance
            pixel_distance = ((vx - px)**2 + (vy - py)**2)**0.5
            
            # Normalized distance: distance relative to average vehicle width
            avg_vehicle_width = v_width
            normalized_distance = pixel_distance / avg_vehicle_width if avg_vehicle_width > 0 else float('inf')
            
            # VERY STRICT: Person must be EXTREMELY close (< 0.8 vehicle widths)
            # AND person must be inside or on the vehicle (not just nearby on street)
            person_in_vehicle_area = py > vy - v_height * 0.3  # Person overlaps vehicle bounds vertically
            
            if normalized_distance < 0.8 and person_in_vehicle_area:
                very_close_person_vehicle_pairs += 1
    
    # Only trigger if MULTIPLE persons are dangerously close to vehicles (actual rescue/emergency scenario)
    return very_close_person_vehicle_pairs >= 2


def analyze_spatial_distribution(vehicles, persons, frame_shape):
    """
    Analyze spatial distribution of detected objects.
    Returns confidence in whether objects are in accident-probable positions.
    Returns 0 if parking scene is detected.
    """
    if len(vehicles) == 0:
        return 0.0
    
    # If parking scene, no accident risk from spatial distribution
    if is_parking_scene(vehicles):
        return 0.0
    
    height, width = frame_shape[0], frame_shape[1]
    confidence_score = 0.0
    
    # Check for objects in lower frame AND with vehicles touching/overlapping
    lower_frame_threshold = height * 0.4
    objects_in_danger_zone = 0
    
    for v in vehicles:
        center_y = (v[1] + v[3]) / 2
        if center_y > lower_frame_threshold:
            objects_in_danger_zone += 1
    
    # Only count if vehicles are actually close together (not just in lower frame)
    if objects_in_danger_zone >= 2:
        # Check for actual close proximity, not just being in same frame region
        close_vehicles = 0
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                norm_dist, _ = calculate_normalized_distance(vehicles[i], vehicles[j])
                # Very close: < 2.0 vehicle widths = potential collision zone
                if norm_dist < 2.0:
                    close_vehicles += 1
        
        if close_vehicles > 0:
            confidence_score += 0.3
    
    # Check for actual collision-risk clustering (very close objects)
    if len(vehicles) >= 2:
        close_pair_count = 0
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                norm_dist, _ = calculate_normalized_distance(vehicles[i], vehicles[j])
                # Objects within 2 vehicle widths = collision risk
                if norm_dist < 2.0:
                    close_pair_count += 1
        
        if close_pair_count > 1:  # Multiple dangerously close pairs
            confidence_score += 0.2
    
    # Presence of persons in danger zone increases confidence
    for p in persons:
        center_y = (p[1] + p[3]) / 2
        if center_y > lower_frame_threshold:
            confidence_score += 0.15
            break
    
    return min(1.0, confidence_score)
# ---------------- SEVERITY ---------------- #

def calculate_severity(vehicle_count, person_count, collision_detected, distance_crash_detected, divider_collision_detected):
    """Improved severity calculation for toll plaza scenarios.

    Adds divider collision as an independent risk factor, since hitting the median/divider
    is a valid accident condition even if no other vehicles are involved.
    """
    severity_score = 0

    # Base score from vehicle count (toll plazas can have multiple vehicles normally)
    if vehicle_count >= 5:
        severity_score += 2
    elif vehicle_count >= 3:
        severity_score += 1

    # Higher weight for actual collision
    if collision_detected:
        severity_score += 3

    # Medium weight for dangerous proximity
    if distance_crash_detected:
        severity_score += 2

    # Divider collision is important and increases risk
    if divider_collision_detected:
        severity_score += 2

    # Persons involved increases severity
    if person_count > 0:
        severity_score += 1

    if severity_score >= 6:
        return "HIGH"
    elif severity_score >= 3:
        return "MEDIUM"
    else:
        return "LOW"

# ---------------- SAVE LOG ---------------- #

def save_log(location,vehicles,persons,severity):

    log = {
        "time":[datetime.datetime.now()],
        "location":[location],
        "vehicles":[vehicles],
        "persons":[persons],
        "severity":[severity]
    }

    df = pd.DataFrame(log)

    if os.path.exists("accident_log.csv"):
        df.to_csv("accident_log.csv",mode="a",header=False,index=False)
    else:
        df.to_csv("accident_log.csv",index=False)

# ---------------- EMERGENCY NOTIFICATIONS ---------------- #

def send_emergency_notifications(location, severity, vehicle_count, person_count):
    """
    Simulate sending emergency notifications to police and hospital
    In a real system, this would integrate with actual emergency services
    """
    notifications = []

    # Police notification
    police_msg = f"🚔 POLICE ALERT: Accident detected at {location}. Severity: {severity}. {vehicle_count} vehicles, {person_count} persons involved."
    notifications.append(("Police Station", police_msg))

    # Hospital notification
    hospital_msg = f"🚑 HOSPITAL ALERT: Accident at {location}. Severity: {severity}. Potential injuries: {person_count} persons. Prepare emergency response."
    notifications.append(("Nearest Hospital", hospital_msg))

    # Traffic Control notification
    traffic_msg = f"🚦 TRAFFIC CONTROL: Road closure recommended at {location}. {vehicle_count} vehicles involved in accident."
    notifications.append(("Traffic Control Center", traffic_msg))

    return notifications

def get_severity_color(severity):
    """Return color and styling for severity levels"""
    if severity == "HIGH":
        return "🔴", "#ff4444", "Critical - Immediate Response Required"
    elif severity == "MEDIUM":
        return "🟡", "#ffaa00", "Moderate - Response Needed"
    else:
        return "🟢", "#44aa44", "Low - Monitor Situation"

# NEW: Collapsed vehicle with messy surroundings detection
def detect_collapsed_vehicle_with_messy_surroundings(vehicles, all_objects, frame_shape):
    """
    Detect a SINGLE severely collapsed/damaged vehicle with messy surroundings.
    
    This catches scenarios where:
    - One vehicle is heavily damaged/deformed
    - Surrounding area shows debris/scattered objects (mess indicators)
    - No multi-vehicle collision, but still critical accident
    
    Examples: Single vehicle rollover, severe crash into infrastructure
    """
    if len(vehicles) == 0:
        return False
    
    # Only handle single or dual vehicle scenarios (not multi-vehicle pile-up)
    if len(vehicles) > 2:
        return False
    
    height, width = frame_shape[0], frame_shape[1]
    frame_area = width * height
    
    # Find the largest/main vehicle
    main_vehicle_idx = 0
    main_area = 0
    for i, v in enumerate(vehicles):
        v_area = max(0, v[2] - v[0]) * max(0, v[3] - v[1])
        if v_area > main_area:
            main_area = v_area
            main_vehicle_idx = i
    
    if main_area == 0:
        return False
    
    main_v = vehicles[main_vehicle_idx]
    x1, y1, x2, y2 = main_v
    w = x2 - x1
    h = y2 - y1 + 1e-9
    aspect_ratio = w / h
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Check 1: Vehicle is severely deformed/collapsed
    severely_deformed = (
        (aspect_ratio > 2.0 or aspect_ratio < 0.45) or  # Extreme deformation
        (main_area >= 0.14 * frame_area and (aspect_ratio > 1.7 or aspect_ratio < 0.58))  # Severe with medium size
    )
    
    if not severely_deformed:
        return False
    
    # Check 2: Vehicle is medium-to-large (significant impact size)
    significant_size = main_area >= 0.10 * frame_area
    
    if not significant_size:
        return False
    
    # Check 3: Surroundings are messy (lots of detections around the vehicle)
    # Count objects detected in the vicinity of the vehicle (within 1.5x the vehicle bounds)
    vehicle_region_x1 = max(0, x1 - w * 0.5)
    vehicle_region_y1 = max(0, y1 - h * 0.5)
    vehicle_region_x2 = min(width, x2 + w * 0.5)
    vehicle_region_y2 = min(height, y2 + h * 0.5)
    
    objects_in_region = 0
    if all_objects is not None and len(all_objects) > 0:
        for obj in all_objects:
            obj_x1, obj_y1, obj_x2, obj_y2 = obj
            obj_center_x = (obj_x1 + obj_x2) / 2
            obj_center_y = (obj_y1 + obj_y2) / 2
            
            # Count objects (excluding main vehicle) in the expanded region
            if (vehicle_region_x1 <= obj_center_x <= vehicle_region_x2 and 
                vehicle_region_y1 <= obj_center_y <= vehicle_region_y2):
                # Don't count the main vehicle itself
                if obj != main_v:
                    objects_in_region += 1
    
    # Check 4: For single vehicle accident, messy surroundings help confirm
    # Multiple objects in vicinity = debris/wreckage
    has_messy_surroundings = objects_in_region >= 2
    
    # ACCIDENT VERDICT: Severely deformed single vehicle + messy surroundings
    if severely_deformed and significant_size and has_messy_surroundings:
        return True
    
    # ALTERNATIVE: Very severely deformed large vehicle (even without multiple surrounding objects)
    # Extreme deformation alone can indicate severe accident
    if (main_area >= 0.16 * frame_area) and (aspect_ratio > 2.2 or aspect_ratio < 0.4):
        return True
    
    # ALTERNATIVE 2: Severely deformed medium vehicle with debris
    if has_messy_surroundings and (aspect_ratio > 2.0 or aspect_ratio < 0.45):
        return True
    
    return False

# ---------------- ACCIDENT DECISION ---------------- #

def accident_surroundings(vehicles, persons, frame_shape):
    """Assess surrounding context to detect collapsed/crash scene.
    
    ONLY triggers for actual multi-vehicle collisions or severe single-vehicle damage.
    Normal street scenes with people near parked vehicles = NOT an accident.
    """
    height, width = frame_shape[0], frame_shape[1]
    frame_area = height * width

    if len(vehicles) == 0:
        return False

    # Only look at this if multiple vehicles present
    # (Single vehicle alone doesn't indicate surrounding accident context)
    if len(vehicles) < 2:
        return False

    # find main large vehicle candidate
    sorted_vehicles = sorted(vehicles, key=lambda v: (v[2]-v[0])*(v[3]-v[1]), reverse=True)
    main = sorted_vehicles[0]
    mx1, my1, mx2, my2 = main
    m_w = mx2 - mx1
    m_h = my2 - my1
    
    main_area = m_w * m_h

    # Count vehicles in TIGHT collision cluster (actual multi-vehicle crash)
    neighbors_in_collision = 0
    for v in vehicles[1:]:
        vx1, vy1, vx2, vy2 = v
        # Normalize distance
        norm_dist, _ = calculate_normalized_distance(main, v)
        # VERY tight: < 1.2 vehicle widths = collision
        if norm_dist < 1.2:
            neighbors_in_collision += 1

    # Multiple vehicles in collision cluster = accident
    if neighbors_in_collision >= 2:
        return True

    # Very large vehicle with abnormal aspect (flipped/rolled) = crash
    m_aspect = m_w / (m_h + 1e-9)
    if main_area >= 0.24 * frame_area and (m_aspect > 1.5 or m_aspect < 0.6):
        return True

    # STRICT: Person must be INSIDE vehicle bounds AND no normality
    # (prevents flagging street scenes with pedestrians)
    for p in persons:
        px = (p[0] + p[2]) / 2
        py = (p[1] + p[3]) / 2
        
        # Only count if person is LITERALLY INSIDE vehicle bounds
        person_inside = (px >= mx1 and px <= mx2) and (py >= my1 and py <= my2)
        
        if person_inside:
            return True

    return False


def detect_heavy_vehicle_infrastructure_collision(vehicles, all_objects, frame_shape):
    """
    Detect heavy vehicles (bus/truck) in collision with infrastructure/equipment.
    
    Scenarios: Bus hit by crane/equipment, truck into building/machinery, etc.
    Key indicators: Large heavy vehicle + multiple objects in any position
    """
    if vehicles is None or len(vehicles) == 0:
        return False
    
    if all_objects is None:
        all_objects = []
    
    height, width = frame_shape[0], frame_shape[1]
    frame_area = width * height
    
    # Heavy vehicle: area >= 0.08 frame (VERY lenient for bus-sized vehicles)
    large_vehicles = []
    for v in vehicles:
        x1, y1, x2, y2 = v
        v_area = max(0, x2 - x1) * max(0, y2 - y1)
        if v_area >= 0.08 * frame_area:  # Very low threshold to catch bus/truck
            large_vehicles.append((v, v_area))
    
    if len(large_vehicles) == 0:
        return False
    
    # For each large vehicle, check if there are additional objects (infrastructure)
    for vehicle, v_area in large_vehicles:
        x1, y1, x2, y2 = vehicle
        w = x2 - x1
        h = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Count nearby objects (wider search region for infrastructure)
        search_x1 = max(0, x1 - w)
        search_y1 = max(0, y1 - h)
        search_x2 = min(width, x2 + w)
        search_y2 = min(height, y2 + h)
        
        nearby_objects = 0
        for obj in all_objects:
            obj_x1, obj_y1, obj_x2, obj_y2 = obj
            obj_center_x = (obj_x1 + obj_x2) / 2
            obj_center_y = (obj_y1 + obj_y2) / 2
            
            if (search_x1 <= obj_center_x <= search_x2 and 
                search_y1 <= obj_center_y <= search_y2):
                if obj != vehicle:  # Different object
                    nearby_objects += 1
        
        # VERY LENIENT: Just 1+ nearby objects + large vehicle = infrastructure collision
        if nearby_objects >= 1:
            return True
        
        # Alternative: Very large vehicle (0.15+ frame) with ANY objects nearby = accident
        # This catches buses even if equipment isn't well-detected
        if v_area >= 0.15 * frame_area and len(all_objects) >= 2:
            return True
    
    return False


def accident_decision(collision, distance_crash, divider_hit, single_impact, heavy_impact, collapsed_wreck, human_factor, parking_scene, surroundings_flag, spatial_confidence=0.0, vehicles=None, persons=None, frame_shape=None, all_objects=None):
    """Robust final accident decision with clear collision priority.
    
    Enhanced to detect:
    - Vehicle-to-vehicle collisions
    - Severe positioning accidents (like bus-equipment collisions)
    - Single collapsed vehicle with messy surroundings  
    - Wreck scenarios
    
    KEY PRINCIPLE: An accident requires ACTUAL VEHICLE COLLISION or clear dangerous proximity.
    Persons near vehicles (street scenes, loading, etc.) do NOT constitute an accident.
    """
    if parking_scene:
        return False
    
    # NEW: Check for severe positioning accidents (like bus-equipment collisions)
    severe_positioning = False
    if vehicles is not None and frame_shape is not None:
        severe_positioning = detect_severe_positioning_accident(vehicles, frame_shape)
    
    # CRITICAL NEW: Check for collapsed single vehicle with messy surroundings
    collapsed_single_vehicle = False
    if vehicles is not None and frame_shape is not None:
        collapsed_single_vehicle = detect_collapsed_vehicle_with_messy_surroundings(vehicles, all_objects, frame_shape)
    
    # CRITICAL NEW: Check for heavy vehicle (bus/truck) in collision with infrastructure/equipment
    heavy_vehicle_collision = False
    if vehicles is not None and all_objects is not None and frame_shape is not None:
        heavy_vehicle_collision = detect_heavy_vehicle_infrastructure_collision(vehicles, all_objects, frame_shape)
    
    # STRONG SIGNALS: Vehicle-to-vehicle collision/damage (primary accident indicators)
    if any([collision, single_impact, heavy_impact, collapsed_wreck, severe_positioning, collapsed_single_vehicle, heavy_vehicle_collision]):
        return True
    
    # MEDIUM SIGNALS: Dangerous vehicle proximity (require high spatial confidence)
    spatial_support = spatial_confidence > 0.3
    if distance_crash and spatial_support:
        return True
    
    # DIVIDER COLLISION: Only with additional evidence
    if divider_hit and any([collision, distance_crash, single_impact, heavy_impact, collapsed_wreck, severe_positioning, collapsed_single_vehicle, heavy_vehicle_collision]):
        return True
    
    # SURROUNDINGS: Only with strong spatial evidence
    if surroundings_flag and spatial_support:
        return True

    # NEW: Emergency risk context fallback based on vehicles+persons closeness
    if persons is not None and accident_human_vehicle_context(vehicles, persons, frame_shape, all_objects):
        return True



def accident_human_vehicle_context(vehicles, persons, frame_shape, all_objects=None):
    """Fallback detection when vehicles + persons indicate accident scene (damaged vehicle + bystanders/rescue)."""
    if len(vehicles) == 0 or len(persons) == 0:
        return False

    # avoid parking lots marked as static
    if is_parking_scene(vehicles):
        return False

    height, width = frame_shape[0], frame_shape[1]
    frame_area = height * width

    # CORE RULE 1: Multiple persons + vehicles present in non-parking context = accident scene
    # This catches: people standing near damaged vehicles, rescue operations, debris scenes
    if len(vehicles) >= 1 and len(persons) >= 2:
        return True

    # CORE RULE 2: ANY person very close to ANY vehicle (< 1.2x vehicle width away)
    # This catches: person inside wreckage, rescue workers near impact zone
    for v in vehicles:
        vx1, vy1, vx2, vy2 = v
        vcx, vcy = (vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0
        v_width = vx2 - vx1
        
        if v_width <= 0:
            continue

        for p in persons:
            px1, py1, px2, py2 = p
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            dist = ((vcx - pcx)**2 + (vcy - pcy)**2)**0.5
            norm = dist / v_width

            if norm < 1.2:  # Within 1.2 vehicle widths = emergency contact zone
                return True

    # CORE RULE 3: Multiple vehicles close to each other (tightly clustered) + any persons = multi-vehicle accident
    if len(vehicles) >= 2 and len(persons) >= 1:
        tight_pairs = 0
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                norm_dist, _ = calculate_normalized_distance(vehicles[i], vehicles[j])
                if norm_dist < 2.0:  # Vehicles close together
                    tight_pairs += 1
        
        if tight_pairs >= 1:
            return True

    # CORE RULE 4: Large vehicle (>0.10 frame area) + persons = potential heavy-vehicle accident
    for v in vehicles:
        x1, y1, x2, y2 = v
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area >= 0.10 * frame_area and len(persons) >= 1:
            return True

    return False

# Page configuration
st.set_page_config(
    page_title="🚨 AI Accident Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .accident-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .safe-status {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .severity-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .severity-medium {
        background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .severity-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .notification-card {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚨 AI-Powered Accident Detection System</h1>
    <p>Advanced Computer Vision for Real-Time Traffic Safety Monitoring</p>
    <p style="font-size: 0.9em; opacity: 0.9;">Toll Plaza & Highway Accident Prevention Technology</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Control Panel")

    st.subheader("📹 Camera Selection")
    camera = st.selectbox(
        "Select Monitoring Camera",
        list(camera_locations.keys()),
        help="Choose the camera location for accident detection"
    )
    location = camera_locations[camera]

    st.subheader("📤 File Upload")
    uploaded_file = st.file_uploader(
        "Upload Video/Image for Analysis",
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
        help="Supported formats: MP4, AVI, MOV, JPG, JPEG, PNG"
    )

    st.subheader("⚙️ System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AI Model", "YOLOv8n", "Active")
    with col2:
        st.metric("Confidence", f"{CONFIDENCE_THRESHOLD*100}%", "Threshold")

    st.subheader("🚨 Emergency Contacts")
    st.info("**Police:** 100\n**Ambulance:** 108\n**Fire:** 101")

# Main content area
if uploaded_file:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🔍 Live Analysis", "📊 Detection Results", "📋 Incident Log"])

    with tab1:
        st.subheader("🔍 Real-Time Accident Analysis")

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())
        path = temp.name

        is_video = uploaded_file.type.startswith("video")

        if is_video:
            st.info("🎬 Processing video stream... Analyzing each frame for accident detection.")

            cap = cv2.VideoCapture(path)
            frame_window = st.empty()
            status_window = st.empty()

            accident_frames = 0
            vehicles_confidences = []
            persons_confidences = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                vehicles = []
                persons = []
                vehicles_confidences = []
                persons_confidences = []

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])

                        if conf < CONFIDENCE_THRESHOLD:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        coords = [x1, y1, x2, y2]

                        if cls in vehicle_classes and area >= MIN_VEHICLE_SIZE:
                            vehicles.append(coords)
                            vehicles_confidences.append(conf)
                        if cls == person_class:
                            persons.append(coords)
                            persons_confidences.append(conf)

                # Collect detected classes
                detected_classes = set()
                all_objects = []  # NEW: Collect all detected objects for collapsed vehicle detector
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            all_objects.append([x1, y1, x2, y2])  # NEW: Store all bounding boxes
                            
                            if cls == 0:
                                detected_classes.add("person")
                            elif cls == 2:
                                detected_classes.add("car")
                            elif cls == 3:
                                detected_classes.add("motorcycle")
                            elif cls == 5:
                                detected_classes.add("bus")
                            elif cls == 7:
                                detected_classes.add("truck")

                # Keep usual legacy flags
                collision = vehicle_collision(vehicles)
                distance_crash = vehicle_distance_crash(vehicles, vehicles_confidences)
                divider_hit = adjusted_divider_collision(vehicles, frame.shape)
                single_impact = single_vehicle_impact(vehicles, frame.shape)
                heavy_impact = heavy_vehicle_impact(vehicles, frame.shape)
                collapsed_wreck = collapsed_wreck_fallback(vehicles, persons, frame.shape)
                human_factor = human_near_vehicle(vehicles, persons, vehicles_confidences, persons_confidences)
                parking_scene = is_parking_scene(vehicles)
                # Analyze spatial distribution to guide detection
                spatial_confidence = analyze_spatial_distribution(vehicles, persons, frame.shape)
                # keep human factor measured but not alone as a collision accident trigger

                # do not trigger collision due to parked arrangement
                if parking_scene:
                    collision = False
                    distance_crash = False
                    divider_hit = False

                surroundings_flag = accident_surroundings(vehicles, persons, frame.shape)
                accident = accident_decision(collision, distance_crash, divider_hit, single_impact, heavy_impact, collapsed_wreck, human_factor, parking_scene, surroundings_flag, spatial_confidence, vehicles, persons, frame.shape, all_objects)

                # Always compute severity for UI consistency
                severity = calculate_severity(len(vehicles), len(persons), collision, distance_crash, divider_hit)

                # Debug output
                status_window.write({
                    'vehicles': vehicles,
                    'persons': persons,
                    'collision': collision,
                    'distance_crash': distance_crash,
                    'divider_hit': divider_hit,
                    'single_impact': single_impact,
                    'heavy_impact': heavy_impact,
                    'collapsed_wreck': collapsed_wreck,
                    'human_factor': human_factor,
                    'spatial_confidence': spatial_confidence,
                    'parking_scene': parking_scene,
                    'surroundings_flag': surroundings_flag,
                    'accident': accident
                })

                if results and len(results) > 0:
                    annotated = results[0].plot()
                    frame_window.image(annotated, channels="BGR", use_column_width=True)
                else:
                    frame_window.image(frame, channels="BGR", use_column_width=True)

                # Status display
                if accident:
                    alert_text = "Emergency response activated"
                    if divider_hit:
                        alert_text = "🧱 Divider collision detected — emergency response activated"

                    status_window.markdown(f"""
                    <div class="accident-alert">
                        <h2>🚨 ACCIDENT DETECTED!</h2>
                        <p>{alert_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Emergency notifications
                    severity = calculate_severity(len(vehicles), len(persons), collision, distance_crash, divider_hit)
                    notifications = send_emergency_notifications(location, severity, len(vehicles), len(persons))

                    for recipient, message in notifications:
                        st.markdown(f"""
                        <div class="notification-card">
                            <strong>{recipient}</strong><br>
                            {message}
                        </div>
                        """, unsafe_allow_html=True)

                    break
                else:
                    status_window.markdown("""
                    <div class="safe-status">
                        <h3>✅ Traffic Normal</h3>
                        <p>No accidents detected</p>
                    </div>
                    """, unsafe_allow_html=True)

            cap.release()

        else:
            st.info("🖼️ Analyzing uploaded image for accident detection...")

            image = cv2.imread(path)
            results = model(image)

            vehicles = []
            persons = []
            vehicles_confidences = []
            persons_confidences = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf < CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    coords = [x1, y1, x2, y2]

                    if cls in vehicle_classes and area >= MIN_VEHICLE_SIZE:
                        vehicles.append(coords)
                        vehicles_confidences.append(conf)
                    if cls == person_class:
                        persons.append(coords)
                        persons_confidences.append(conf)

            # Collect detected classes
            detected_classes = set()
            all_objects = []  # NEW: Collect all detected objects for collapsed vehicle detector
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        all_objects.append([x1, y1, x2, y2])  # NEW: Store all bounding boxes
                        
                        if cls == 0:
                            detected_classes.add("person")
                        elif cls == 2:
                            detected_classes.add("car")
                        elif cls == 3:
                            detected_classes.add("motorcycle")
                        elif cls == 5:
                            detected_classes.add("bus")
                        elif cls == 7:
                            detected_classes.add("truck")

            # Prepare vehicles as list of dicts for scoring
            # Keep usual legacy flags
            collision = vehicle_collision(vehicles)
            distance_crash = vehicle_distance_crash(vehicles, vehicles_confidences)
            divider_hit = adjusted_divider_collision(vehicles, image.shape)
            single_impact = single_vehicle_impact(vehicles, image.shape)
            heavy_impact = heavy_vehicle_impact(vehicles, image.shape)
            collapsed_wreck = collapsed_wreck_fallback(vehicles, persons, image.shape)
            human_factor = human_near_vehicle(vehicles, persons, vehicles_confidences, persons_confidences)
            parking_scene = is_parking_scene(vehicles)
            # Analyze spatial distribution to guide detection
            spatial_confidence = analyze_spatial_distribution(vehicles, persons, image.shape)
            # human presence increases severity and notification urgency but doesn't alone set accident state

            # do not trigger collision due to parked arrangement
            if parking_scene:
                collision = False
                distance_crash = False
                divider_hit = False

            surroundings_flag = accident_surroundings(vehicles, persons, image.shape)
            accident = accident_decision(collision, distance_crash, divider_hit, single_impact, heavy_impact, collapsed_wreck, human_factor, parking_scene, surroundings_flag, spatial_confidence, vehicles, persons, image.shape, all_objects)

            severity = calculate_severity(len(vehicles), len(persons), collision, distance_crash, divider_hit)

            # Debug output
            st.write({
                'vehicles': vehicles,
                'persons': persons,
                'collision': collision,
                'distance_crash': distance_crash,
                'divider_hit': divider_hit,
                'single_impact': single_impact,
                'heavy_impact': heavy_impact,
                'collapsed_wreck': collapsed_wreck,
                'human_factor': human_factor,
                'spatial_confidence': spatial_confidence,
                'parking_scene': parking_scene,
                'all_objects_count': len(all_objects) if all_objects else 0,
                'accident': accident
            })

            if results and len(results) > 0:
                annotated = results[0].plot()
                st.image(annotated, channels="BGR", use_column_width=True, caption="AI Analysis Results")
            else:
                st.image(image, channels="BGR", use_column_width=True, caption="Original Image - No detections")

            if accident:
                severity = calculate_severity(len(vehicles), len(persons), collision, distance_crash, divider_hit)
                severity_icon, severity_color, severity_desc = get_severity_color(severity)

                st.markdown(f"""
                <div class="accident-alert">
                    <h2>🚨 ACCIDENT DETECTED!</h2>
                    <p>Immediate emergency response required</p>
                </div>
                """, unsafe_allow_html=True)

                if divider_hit:
                    st.warning("🧱 Divider collision detected")

                # Emergency notifications
                notifications = send_emergency_notifications(location, severity, len(vehicles), len(persons))

                st.subheader("🚨 Emergency Notifications Sent")
                for recipient, message in notifications:
                    st.markdown(f"""
                    <div class="notification-card">
                        <strong>{recipient}</strong><br>
                        {message}
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="safe-status">
                    <h2>✅ No Accident Detected</h2>
                    <p>Traffic conditions are normal</p>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📊 Detailed Detection Results")

        if 'accident' in locals():
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🚗 Vehicles Detected", len(vehicles))
            with col2:
                st.metric("🧍 Persons Detected", len(persons))
            with col3:
                severity_icon, severity_color, severity_desc = get_severity_color(severity)
                st.metric("⚠️ Severity Level", f"{severity_icon} {severity}", severity_desc)

            # Detection details
            st.subheader("🔍 Detection Analysis")

            detection_data = {
                "Detection Type": ["Vehicle Collision", "Dangerous Proximity", "Divider Collision", "Human Involvement"],
                "Status": [
                    "✅ Detected" if collision else "❌ Not Detected",
                    "✅ Detected" if distance_crash else "❌ Not Detected",
                    "✅ Detected" if divider_hit else "❌ Not Detected",
                    "✅ Detected" if human_factor else "❌ Not Detected"
                ],
                "Risk Level": [
                    "High" if collision else "None",
                    "Medium" if distance_crash else "None",
                    "Medium" if divider_hit else "None",
                    "High" if human_factor else "None"
                ]
            }

            st.table(pd.DataFrame(detection_data))

            # Severity breakdown
            st.subheader("📈 Severity Assessment")
            severity_icon, severity_color, severity_desc = get_severity_color(severity)

            if severity == "HIGH":
                st.markdown(f"""
                <div class="severity-high">
                    <h3>{severity_icon} HIGH SEVERITY</h3>
                    <p>{severity_desc}</p>
                    <ul>
                        <li>🚨 Immediate emergency response required</li>
                        <li>🚔 Police and ambulance dispatched</li>
                        <li>🚧 Road closure recommended</li>
                        <li>🏥 Medical teams on standby</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif severity == "MEDIUM":
                st.markdown(f"""
                <div class="severity-medium">
                    <h3>{severity_icon} MEDIUM SEVERITY</h3>
                    <p>{severity_desc}</p>
                    <ul>
                        <li>🚨 Emergency response needed</li>
                        <li>🚔 Police notification sent</li>
                        <li>🚑 Ambulance on alert</li>
                        <li>⚠️ Traffic control activated</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="severity-low">
                    <h3>{severity_icon} LOW SEVERITY</h3>
                    <p>{severity_desc}</p>
                    <ul>
                        <li>👁️ Situation monitored</li>
                        <li>📝 Incident logged</li>
                        <li>🚦 Traffic flow maintained</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Save incident
            if st.button("💾 Save Incident Report", type="primary"):
                save_log(location, len(vehicles), len(persons), severity)
                st.success("✅ Incident report saved successfully!")

        else:
            st.info("📤 Upload a file to see detection results")

    with tab3:
        st.subheader("📋 Accident History & Analytics")

        if os.path.exists("accident_log.csv"):
            df = pd.read_csv("accident_log.csv")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Incidents", len(df))
            with col2:
                st.metric("High Severity", len(df[df["severity"] == "HIGH"]))
            with col3:
                st.metric("Medium Severity", len(df[df["severity"] == "MEDIUM"]))
            with col4:
                st.metric("Low Severity", len(df[df["severity"] == "LOW"]))

            # Severity distribution chart
            st.subheader("📊 Severity Distribution")
            severity_counts = df["severity"].value_counts()
            st.bar_chart(severity_counts)

            # Recent incidents table
            st.subheader("🕐 Recent Incidents")
            st.dataframe(df.tail(10), use_container_width=True)

            # Location analysis
            st.subheader("📍 Incident Locations")
            location_counts = df["location"].value_counts()
            st.bar_chart(location_counts)

        else:
            st.info("📝 No incident history available yet")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Welcome to AI Accident Detection System</h2>
        <p style="font-size: 1.2em; color: #666;">
            Upload a video or image to begin real-time accident detection and analysis
        </p>
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; margin: 2rem 0;">
            <h3>🎯 System Capabilities</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4>🚗 Vehicle Detection</h4>
                    <p>Advanced AI detects cars, bikes, buses, and trucks with high accuracy</p>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4>⚠️ Collision Analysis</h4>
                    <p>Real-time collision detection using computer vision algorithms</p>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4>🚨 Emergency Response</h4>
                    <p>Automatic notifications to police, hospitals, and traffic control</p>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4>📊 Severity Assessment</h4>
                    <p>Three-level severity classification for appropriate response</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)