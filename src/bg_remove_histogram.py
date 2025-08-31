import cv2
import numpy as np
from glob import glob
from loguru import logger

def remove_red(frame):
    frame[:,:,2] = np.zeros([frame.shape[0], frame.shape[1]])
    return frame

def erode_dilate(frame):
    logger.info("Dilation and Erosion")
    # Define the structuring element (kernel) for erosion and dilation
    kernel = np.ones((3, 3), np.uint8)  # 5x5 square kernel

    # Perform erosion
    frame = cv2.erode(frame, kernel, iterations=1)

    # Perform dilation
    frame = cv2.dilate(frame, kernel, iterations=1)

    return frame
    
# Function to remove dominant colors based on histograms
def remove_dominant_colors(frame, add_mask_4ch=True):

    logger.info("Removing Pixels With Dominant Colors...")
    # Convert the image to a different color space if needed (e.g., BGR to HSV)
    # Here, we convert to HSV to work with color components
    # Define the kernel or structuring element (a square in this case)

    _frame = frame.copy()
    #kernel = np.ones((3, 3), np.uint8)
    # Perform erosion on the input image
    #erod_frame = cv2.erode(erod_frame, kernel, iterations=10)
    #cv2.imshow('Processed Frame', erod_frame)
    #cv2.waitKey(1000)
   
    _frame = remove_red(_frame) # removing red color hoping this would increase contrast between vegetation and forground
    
    hsv_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
    
    del _frame

    # Calculate color histograms for the frame
    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    # Sort the histogram matrix postion based on 
    # the hist value and make it 1D (axis=None)
    hist_pos = np.argsort(hist, axis=None)
    # Sort histogram by value (number of pixels for each color)
    ordered_his = np.sort(hist.ravel())
    # Exclude the histogram postions with 0 pixels
    hist_pos = hist_pos[ordered_his!=0.0]

    # Exclude the values postions with 0 pixels
    ordered_his = ordered_his[ordered_his!=0]

    h,w = frame.shape[:-1] # number of pixels in the frame
    pixel_majority = int(h * w * 0.7) # How many pixels to remove based on hist value
    pixel_count = 0
    for i,p in enumerate(ordered_his[::-1]):
        pixel_count+=p
        if pixel_count >= pixel_majority: break
    hist_pos = hist_pos[:i*-1]     


    ordered_colors = np.array(np.unravel_index(hist_pos, hist.shape)).transpose()
    size_of_colors = len(ordered_colors)
    colors_to_keep = ordered_colors

    hsv_frame_flat = hsv_frame.reshape((-1,3), order='C')
    frame_flat = frame.reshape((-1,3), order='C')

    color_dic = {tuple(x):False for x in colors_to_keep}

    mask = np.full(frame_flat.shape[:-1], 255, dtype=np.uint8)

    for i, pix in enumerate(hsv_frame_flat):
        remove_pixel = color_dic.get(tuple(pix), True)
        if add_mask_4ch:
            mask[i] = 0 if remove_pixel else 255
        else:
            if remove_pixel:
                frame_flat[i] = [255,255,255]

    if add_mask_4ch:
        frame_flat = np.column_stack((frame_flat, mask))
        frame = frame_flat.reshape((h,w,4), order='C')
    frame = erode_dilate(frame)

    return frame

def display_frame(frame,mask=None, scale=1, visual="merge_avrg"):
    # Display the processed frame

    if visual=="only_removed_pix": # removed pixels will appear and the rest will be red
        condition = np.all(mask != [255,255,255], axis=-1)
        frame[condition] = [0,0,255]
        img = frame
    elif visual=="merge_avrg":
        frame_cpy = frame.copy()
        condition = np.all(mask != [255,255,255], axis=-1)
        frame[condition] = [0,0,255]
        img = cv2.addWeighted(frame, 0.5, frame_cpy, 0.5, 0)
    elif visual=="side_by_side":
        img = np.concatenate((
                                cv2.resize(frame, (0,0), fx=scale, fy=scale), 
                                cv2.resize(mask, (0,0), fx=scale, fy=scale)
                             )
                              , axis=1
                 )
    elif visual=="just_show":
        img = frame
        

    cv2.imshow('Processed Frame', img)
    cv2.waitKey(1000)
    #cv2.destroyAllWindows()

    
# Function to process and display frames one by one
def process_and_display_frames(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        if not ret:
            break

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

        # Remove dominant colors
        processed_frame = remove_dominant_colors(frame.copy())
        #processed_frame = frame
        display_frame(frame, processed_frame, visual="side_by_side")


    # Release video object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def load_images(data_dir):
    for img_name in glob('/'.join([data_dir,"*.jpg"])):
        logger.info(f"Removing main Colors from imgage: {img_name}")
        frame = cv2.imread(img_name)
        processed_frame = remove_dominant_colors(frame.copy())
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

        #display_frame(processed_frame, visual= "just_show")
        # Merge the grayscale image with the BGR image to create a 4-channel image
        processed_frame = cv2.merge((frame[:,:,0], frame[:,:,1], frame[:,:,2], processed_frame))
        #processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)

        logger.info("Saving File with 4 channels...")
        logger.info(f"Saving File: {data_dir}/{img_name[:-4]}_4chn.png")
        # Save the 4-channel image
        cv2.imwrite(f"./{img_name[:-4]}_4chn.png", processed_frame)



def main():
    # Specify the input video file and dominant color threshold
    input_video_path = '/Users/a.e./MEGAsync/Archive/ndsu_BirdsDroneFootage/Video_Examples/009_W_FS_A2_Video_Trim.mp4'

    # Call the function to process and display frames
    #process_and_display_frames(input_video_path)
    load_images("../../finalized_dataset/")

if __name__=="__main__":
    main()

