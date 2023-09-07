import sys
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

first = 0
list_of_topics = ["/accel/filtered",
                  "/joystick/accelerator_cmd",
                  '/joystick/brake_cmd',
                  "/joystick/gear_cmd",
                  "/joystick/steering_cmd", 
                  '/novatel_bottom/rawimu',
                  '/novatel_top/rawimu',
                  '/raptor_dbw_interface/accelerator_pedal_report',
                  '/raptor_dbw_interface/brake_2_report',
                  '/raptor_dbw_interface/motec_report',
                  '/raptor_dbw_interface/pt_report',
                  '/raptor_dbw_interface/tire_report',
                  '/raptor_dbw_interface/wheel_speed_report'
                  ]
# file_data = pd.DataFrame(columns=list_of_topics + ["rel_time"] +['abs_time'])
# print(file_data)
file_data = {key: [] for key in list_of_topics}

# Reading from a MCAP file
from mcap_ros2.reader import read_ros2_messages

for msg in read_ros2_messages(sys.argv[1], topics=list_of_topics):
    # print(f"{msg.channel.topic}: f{msg.ros_msg}")
    file_data[msg.channel.topic].append(msg.ros_msg)

print(file_data["/accel/filtered"])

# with open(sys.argv[1], "rb") as f:
#     reader = make_reader(f, decoder_factories=[DecoderFactory()])
#     for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=list_of_topics):
#         if first == 0:
#             first = message.log_time
#         file_data['rel_time']
#         # print(f'{(message.log_time - first) * 1e-9}')
#         print(f"{schema.name} [{message.log_time}]: {ros_msg}")
