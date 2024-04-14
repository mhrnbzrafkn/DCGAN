import cv2

def extract_frames(video_path, output_folder, resolution=(256, 256)):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # Create a VideoWriter object to save images
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_resolution = (resolution[0], resolution[1])
    out = cv2.VideoWriter('output.avi', fourcc, fps, output_resolution)
    
    # Loop through frames and extract
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Resize frame to specified resolution
        frame = cv2.resize(frame, output_resolution)
        
        # Save frame as an image
        frame_count += 1
        filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        
        # Display progress
        print(f"Extracting frame {frame_count}")
        
        # Write frame to output video
        out.write(frame)
    
    # Release video capture and writer objects
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
video_path = './training-src/training_images/video.mp4'
output_folder = './training-src/training_images/frames_output'
resolution = (256, 256)  # Set the desired resolution here
extract_frames(video_path, output_folder, resolution)
