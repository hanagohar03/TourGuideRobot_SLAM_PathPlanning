Update 20.11.2025

Moved old maps to Archive folder, uploaded new maps and world. the pgm had to be zipped due to size limit on github.



The folder VM contains the ressources used to run a simulation in gazebo/rviz.

`GIU_map2.pgm` is a modified version of the original GUI map, the wall were just drawn thicker.

That map was converted into a useable wolrd file for gazebo called `slam_world.sdf`.
The converter `pgm_to_sdf.py` was copied from https://github.com/Arseni1919/Converter_PGM_to_SDF?tab=readme-ov-file

Using SLAM a small section of that map was rescanned and saved to `Testmap.pgm`.

For all maps their corresponding .yaml is also needed.

The script `GIU_world.launch.py` is a modified version of the turtlebot3 world launch file with just the world changed.
It should be moved to `/opt/ros/humble/share/turtlebot3_gazebo/launch/`.
You will need to edit the file, line 36 is the world location.

Gazebo can then be lauched from a terminal with:
`ros2 launch turtlebot3_gazebo GIU_world.launch.py`

From the folder of the map location a terminal has to opened and with the command:
`ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True map:=Testmap.yaml`

rviz is launched with the nav2 interface.

From the bar at the top select "2D Pose Estimate" set the robot pose. A Pointcloud around the robot should appear and the obstacles inflated.

If you run into problems, like the map not working correctly, consider this youtube video: https://www.youtube.com/watch?v=idQb2pB-h2Q&t=107s.
Around 36 Minutes there is a section about communication fixes.

If you want to control the robots, you need the teleop package, installed via
`sudo apt install ros-humble-teleop-twist-keyboard`

then run
`ros2 run turtlebot3_teleop teleop_keyboard`
in a third terminal.

