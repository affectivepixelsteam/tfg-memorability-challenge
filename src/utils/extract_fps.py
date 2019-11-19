import cv2
import os, time, multiprocessing
import pandas as pd
from functools import partial

def getDatabaseSplitted() :
    # Path
    splitted_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/ground-truth_dev-set_splitted.csv'
    
    # load dataframes
    df_ground_truth = pd.read_csv(splitted_ground_truth_file_path)

    return df_ground_truth['video'].values

def extract_FPS_parallel(input_video_path, output_path):
    list_videos = getDatabaseSplitted()

    start_time = time.time()
    pool = multiprocessing.Pool()  # processes = 7
    FPS_extractor = partial(extract_FPS_from_video,  # get_npy_with_previous_clip_AVEC2019
                                              input_video_path=input_video_path,
                                              output_path=output_path)
    pool.map(FPS_extractor, list_videos)
    pool.close()
    pool.join()
    final_time = (time.time() - start_time)
    print("--- %s Data preparation TIME IN min ---" % (final_time / 60))

def extract_FPS_from_video(video_name, input_video_path, output_path, format_img = ".png"):
    print("Start frames: ", video_name)
    
    prev_name = video_name.split(".")[0]
    new_dir = os.path.join(output_path, prev_name)

    if(not os.path.isdir(new_dir)):
        os.makedirs(new_dir)

        vidcap = cv2.VideoCapture(os.path.join(input_video_path, video_name))
        success,image = vidcap.read()
        count = 0
    
        while success:
            new_name = prev_name+"_"+str(count)
            resized_image = cv2.resize(image,(int(img_width),int(img_heigh)))
            img_output_path = os.path.join(new_dir,new_name+format_img)
            cv2.imwrite(img_output_path, resized_image)     # save frame as JPEG file
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1

    print("End frames: ", video_name)
    
input_video_path = "../../data/corpus/devset/dev-set/sources"
output_path = "/media/marcoscollado/gth10b/tfg-memorabilty-challenge"
img_width = 640
img_heigh = 480
extract_FPS_parallel(input_video_path, output_path)