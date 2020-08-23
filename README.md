# UMIC_TEAM4-Final
The Final Setup for our UMIC recruitment project

=> Latest Update
  ball_detection works,
  the commented return part returns the centre coordinates of the ball wrt to the 400x400 frame the camera captures
  
  to run,
  
  clone repo,
  make workspace with an empty src,
  catkin build,
  go to CMAKE of final_description and comment the .py scripts inclusion part,
  replace empty src with required source,
  catkin build again
  (you shouldnt get any errors, just a warning)
  roslaunch final_description controls.launch
  in new tab-
  rosrun final_description ball_detection.py
  
  (remember to chmod the py scripts and uncomment them from the CMAke...)
  
