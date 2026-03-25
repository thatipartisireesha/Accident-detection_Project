#!/usr/bin/env python3
"""
Direct test of accident detection on accident1.jpg using app.py logic
"""
import sys
sys.path.insert(0, 'd:\\FINAL PROJECT')

from ultralytics import YOLO
import cv2
import tempfile

# Import actual functions from app.py (but we need to be careful about imports)
# Instead, let's load the functions inline from app.py

# Load model and image
print("Loading model and image...")
model = YOLO('yolov8n.pt')
img = cv2.imread('data/accident1.jpg')
if img is None:
    print("ERROR: Could not load image!")
    sys.exit(1)

res = model(img, conf=0.5)
print(f"✓ Detections: {len(res[0].boxes)} boxes")

# Configuration (from app.py)
vehicle_classes = [2, 3, 5, 7]
person_class = 0
CONFIDENCE_THRESHOLD = 0.5
MIN_VEHICLE_SIZE = 2000

vehicles = []
persons = []

# Extract vehicles and persons (app.py logic)
for box in res[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = [float(x) for x in box.xyxy[0]]
    area = (x2 - x1) * (y2 - y1)
    
    if conf < CONFIDENCE_THRESHOLD:
        continue
    
    coords = [x1, y1, x2, y2]
    
    if cls in vehicle_classes and area >= MIN_VEHICLE_SIZE:
        vehicles.append(coords)
    elif cls == person_class:
        persons.append(coords)

print(f"✓ Vehicles: {len(vehicles)}, Persons: {len(persons)}")

# Core test: accident_human_vehicle_context logic
print("\n" + "="*60)
print("Testing accident_human_vehicle_context logic:")
print("="*60)

height, width = img.shape[0], img.shape[1]
frame_area = height * width

print(f"Frame shape: {img.shape}, area: {frame_area}")

# Rule 1: 1+ vehicles AND 2+ persons
rule1 = len(vehicles) >= 1 and len(persons) >= 2
print(f"Rule 1 (1+ vehicles AND 2+ persons): {rule1}")
if rule1:
    print("  ✓ RULE 1 TRIGGERED - ACCIDENT DETECTED!")

# Rule 2: Person within 1.2 vehicle-widths
rule2 = False
if len(vehicles) > 0 and len(persons) > 0:
    for v in vehicles:
        vx1, vy1, vx2, vy2 = v
        vcx, vcy = (vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0
        v_width = vx2 - vx1
        
        for p in persons:
            px1, py1, px2, py2 = p
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            dist = ((vcx - pcx)**2 + (vcy - pcy)**2)**0.5
            norm = dist / v_width if v_width > 0 else float('inf')
            
            if norm < 1.2:
                rule2 = True
                print(f"Rule 2 (person at norm={norm:.2f} < 1.2): {rule2}")
                print("  ✓ RULE 2 TRIGGERED - ACCIDENT DETECTED!")
                break
        if rule2:
            break

# Rule 4: Large vehicle + persons
rule4 = False
if len(vehicles) > 0 and len(persons) > 0:
    for v in vehicles:
        x1, y1, x2, y2 = v
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area >= 0.10 * frame_area:
            rule4 = True
            print(f"Rule 4 (large vehicle area={area:.0f} >= 0.10*frame): {rule4}")
            print("  ✓ RULE 4 TRIGGERED - ACCIDENT DETECTED!")
            break

# FINAL RESULT
print("\n" + "="*60)
accident_detected = rule1 or rule2 or rule4
if accident_detected:
    print("🚨 ACCIDENT DETECTED! 🚨")
else:
    print("✅ No accident detected")
print("="*60)

print(f"\nTest Result Summary:")
print(f"  Vehicles found: {len(vehicles)}")
print(f"  Persons found: {len(persons)}")
print(f"  Rule 1 (vehicles+persons): {rule1}")
print(f"  Rule 2 (close proximity): {rule2}")
print(f"  Rule 4 (large vehicle): {rule4}")
print(f"  Final: {accident_detected}")
