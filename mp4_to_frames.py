import cv2
import os

folder = './movies'
fname = 'NationsLeagueDepay2022.mp4'
path_to_file = os.path.join(folder, fname)
# filename = './movies/penalty.mp4'
vidcap = cv2.VideoCapture(path_to_file)
success,image = vidcap.read()
count = 0

while success:
  cv2.imwrite(f"./movies/frames/input/{fname}_frame{count:05}.jpg", image)     # save frame as JPEG file, zero pad frames      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1