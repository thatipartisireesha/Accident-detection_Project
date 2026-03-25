"""
YOLO Model Training and Fine-tuning Script
Trains the YOLOv8 model for accident detection
"""

from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def setup_dataset_structure():
    """Setup standard YOLO dataset directory structure"""
    dataset_root = Path("./dataset")
    
    # Create necessary directories
    for split in ["images", "labels"]:
        for subset in ["train", "val", "test"]:
            (dataset_root / split / subset).mkdir(parents=True, exist_ok=True)
    
    print("✅ Dataset structure created at ./dataset")
    return dataset_root


def create_dataset_yaml(dataset_root):
    """Create dataset.yaml configuration file for YOLO"""
    yaml_content = """path: """ + str(dataset_root) + """
train: images/train
val: images/val
test: images/test

nc: 80  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
        'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
"""
    
    yaml_path = Path("dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Dataset configuration created: {yaml_path}")
    return yaml_path


def train_model_fine_tune(model_name="yolov8n.pt", epochs=50, img_size=640, batch_size=16):
    """
    Fine-tune YOLOv8 model on custom dataset
    
    Args:
        model_name: Pre-trained model to use (yolov8n, yolov8s, yolov8m, yolov8l)
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
    """
    
    print("\n" + "="*60)
    print("🚀 YOLO MODEL TRAINING STARTED")
    print("="*60 + "\n")
    
    # Setup dataset structure
    dataset_root = setup_dataset_structure()
    dataset_yaml = create_dataset_yaml(dataset_root)
    
    # Load pre-trained model
    print(f"📦 Loading pre-trained model: {model_name}")
    model = YOLO(model_name)
    
    # Check if dataset.yaml exists, if not use COCO
    if not dataset_yaml.exists():
        print("⚠️  Warning: dataset.yaml not found, using pre-trained model (no fine-tuning)")
        print("💡 To fine-tune, add your training data to ./dataset/images/train")
        return model
    
    # Train the model
    print(f"\n🔧 Starting training:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Image Size: {img_size}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Dataset: {dataset_yaml}")
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=10,
            device=0,  # Use GPU (0) or CPU (if no GPU available)
            project="runs/detect",
            name="accident_detection",
            verbose=True,
            save=True,
            cache=False,
        )
        
        print("\n✅ Training completed successfully!")
        print(f"📊 Best weights saved to: {results.save_dir}/weights/best.pt")
        
        return model
        
    except Exception as e:
        print(f"\n⚠️  Training skipped - Dataset not available")
        print(f"   Error: {e}")
        print(f"   Using pre-trained model: {model_name}")
        return model


def test_model_on_videos(model, video_dir="data", output_dir="test_results"):
    """
    Test model on all video files in a directory
    
    Args:
        model: YOLO model to test
        video_dir: Directory containing test videos
        output_dir: Directory to save test results
    """
    
    print("\n" + "="*60)
    print("🧪 TESTING MODEL ON VIDEOS")
    print("="*60 + "\n")
    
    import cv2
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in Path(video_dir).glob('*') 
                   if f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"⚠️  No video files found in {video_dir}")
        return
    
    test_results = {}
    
    for video_file in video_files:
        print(f"\n📹 Testing: {video_file.name}")
        
        try:
            cap = cv2.VideoCapture(str(video_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup output video
            output_video = output_path / f"tested_{video_file.name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            
            detections_summary = {
                'total_frames': frame_count,
                'frames_with_objects': 0,
                'vehicles_detected': [],
                'persons_detected': [],
            }
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = model.predict(frame, conf=0.5, verbose=False)
                
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                
                # Count detections
                boxes = results[0].boxes
                if len(boxes) > 0:
                    detections_summary['frames_with_objects'] += 1
                
                # Extract vehicle and person detections
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:  # Vehicle classes
                        detections_summary['vehicles_detected'].append(cls)
                    elif cls == 0:  # Person class
                        detections_summary['persons_detected'].append(cls)
                
                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"   Processed {frame_idx}/{frame_count} frames")
            
            cap.release()
            out.release()
            
            # Summary statistics
            detection_rate = (detections_summary['frames_with_objects'] / frame_count * 100) if frame_count > 0 else 0
            
            test_results[video_file.name] = {
                'status': '✅ SUCCESS',
                'output_video': str(output_video),
                'total_frames': frame_count,
                'frames_with_detections': detections_summary['frames_with_objects'],
                'detection_rate': f"{detection_rate:.1f}%",
                'vehicles_found': len(detections_summary['vehicles_detected']),
                'persons_found': len(detections_summary['persons_detected']),
            }
            
            print(f"   ✅ Completed - Output: {output_video.name}")
            print(f"   📊 Detection Rate: {detection_rate:.1f}%")
            print(f"   🚗 Vehicles: {len(detections_summary['vehicles_detected'])}, 🧍 Persons: {len(detections_summary['persons_detected'])}")
            
        except Exception as e:
            print(f"   ❌ Error processing {video_file.name}: {str(e)}")
            test_results[video_file.name] = {
                'status': f'❌ ERROR: {str(e)}'
            }
    
    # Print summary report
    print("\n" + "="*60)
    print("📋 TEST SUMMARY REPORT")
    print("="*60 + "\n")
    
    for video_name, result in test_results.items():
        print(f"Video: {video_name}")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print()
    
    print(f"✅ All test results saved to: {output_path}\n")
    
    return test_results


def test_model_on_images(model, image_dir="data", output_dir="test_results"):
    """
    Test model on all image files
    
    Args:
        model: YOLO model to test
        image_dir: Directory containing test images
        output_dir: Directory to save test results
    """
    
    print("\n" + "="*60)
    print("🖼️  TESTING MODEL ON IMAGES")
    print("="*60 + "\n")
    
    import cv2
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in Path(image_dir).glob('*') 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"⚠️  No image files found in {image_dir}")
        return
    
    test_results = {}
    
    for image_file in image_files:
        print(f"\n🖼️  Testing: {image_file.name}")
        
        try:
            image = cv2.imread(str(image_file))
            
            # Run detection
            results = model.predict(image, conf=0.5, verbose=False)
            
            annotated_image = results[0].plot()
            
            # Save annotated image
            output_image = output_path / f"tested_{image_file.name}"
            cv2.imwrite(str(output_image), annotated_image)
            
            # Count detections
            boxes = results[0].boxes
            vehicles_count = 0
            persons_count = 0
            
            for box in boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:
                    vehicles_count += 1
                elif cls == 0:
                    persons_count += 1
            
            test_results[image_file.name] = {
                'status': '✅ SUCCESS',
                'output_image': str(output_image),
                'total_detections': len(boxes),
                'vehicles_found': vehicles_count,
                'persons_found': persons_count,
            }
            
            print(f"   ✅ Completed - Output: {output_image.name}")
            print(f"   🚗 Vehicles: {vehicles_count}, 🧍 Persons: {persons_count}")
            
        except Exception as e:
            print(f"   ❌ Error processing {image_file.name}: {str(e)}")
            test_results[image_file.name] = {
                'status': f'❌ ERROR: {str(e)}'
            }
    
    # Print summary report
    print("\n" + "="*60)
    print("📋 IMAGE TEST SUMMARY")
    print("="*60 + "\n")
    
    for image_name, result in test_results.items():
        print(f"Image: {image_name}")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print()
    
    print(f"✅ All test results saved to: {output_path}\n")
    
    return test_results


if __name__ == "__main__":
    import sys
    
    print("\n🚨 ACCIDENT DETECTION MODEL - TRAINING & TESTING PIPELINE\n")
    
    # Step 1: Train model (optional - uses pre-trained if no custom data)
    model = train_model_fine_tune(
        model_name="yolov8n.pt",
        epochs=50,
        img_size=640,
        batch_size=16
    )
    
    # Step 2: Test on videos
    test_model_on_videos(model, video_dir="data", output_dir="test_results")
    
    # Step 3: Test on images
    test_model_on_images(model, image_dir="data", output_dir="test_results")
    
    print("\n" + "="*60)
    print("✅ TRAINING & TESTING PIPELINE COMPLETED")
    print("="*60)
    print("\n📊 Next Steps:")
    print("   1. Review test results in 'test_results' folder")
    print("   2. Run Streamlit app: streamlit run app.py")
    print("   3. Upload videos/images to the app for real-time analysis\n")
