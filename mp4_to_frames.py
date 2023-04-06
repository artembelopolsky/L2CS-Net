import cv2
# filename = './movies/Nations LeagueDepay2022.mp4'
filename = './movies/penalty.mp4'
vidcap = cv2.VideoCapture(filename)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(f"./movies/frames/input/frame{count:05}.jpg", image)     # save frame as JPEG file, zero pad frames      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
