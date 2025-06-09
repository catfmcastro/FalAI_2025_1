import cv2
import time

def check_camera(camera_id=0, attempts=3):
    """Check if camera with given ID is available and working."""
    print(f"Attempting to access camera {camera_id}...")
    
    for i in range(attempts):
        try:
            # Try to open the camera
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print(f"Attempt {i+1}/{attempts}: Failed to open camera.")
                time.sleep(1)
                continue
                
            # Try to read a frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"Attempt {i+1}/{attempts}: Camera opened but failed to grab frame.")
                cap.release()
                time.sleep(1)
                continue
                
            # If we got here, the camera is working
            print(f"Success! Camera {camera_id} is working.")
            print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
            
            # Release the camera
            cap.release()
            return True
            
        except Exception as e:
            print(f"Attempt {i+1}/{attempts}: Error accessing camera: {str(e)}")
            time.sleep(1)
            
    print(f"Failed to access camera {camera_id} after {attempts} attempts.")
    return False

def main():
    """Main function to check camera functionality."""
    print("Camera detection tool for Orange Pi")
    print("-----------------------------------")
    
    # Check the default camera (usually 0)
    if check_camera(0):
        print("Camera is functioning properly!")
    else:
        print("Camera is not working or not connected.")
        
        # Try alternative camera IDs
        print("\nTrying alternative camera IDs...")
        for camera_id in [1, 2, -1]:
            if check_camera(camera_id, attempts=1):
                print(f"Found working camera at ID {camera_id}")
                break
        else:
            print("No working cameras found.")

if __name__ == "__main__":
    main()
