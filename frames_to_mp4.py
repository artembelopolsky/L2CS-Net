import cv2
import os

# image_folder = '../movies/out_penalty'
# video_name = '../movies/penalty_best.mp4'

image_folder = './movies/frames/output'
video_name = './movies/penalty_out.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

#video = cv2.VideoWriter(video_name, 0, 30, (width,height))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 15.0, (width,height))


for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    

cv2.destroyAllWindows()
video.release()
