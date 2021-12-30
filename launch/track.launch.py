import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        launch_ros.actions.Node(
            package='robotics1_signment',
            executable='tracker',
            name='tracker'
        ),
    ])
