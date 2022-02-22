# sequence length for data collection
sequence_length = 72

# object points
scenarios_object_points = {'straight': [(145, 90), (130, 90), (130, 72), (145, 72)],
                           'left_turn': [(252, 5), (236, 5), (236, 20), (252, 20)],
                           'right_turn': [(14.171737744295465, -28.644126878945876),
                                          (32.663034952274376, -28.644126878945876),
                                          (32.663034952274376, -13.58142654032553),
                                          (14.171737744295465, -13.58142654032553)]
                           }
# seeds
scenarios_seeds = {'straight': 2, 'left_turn': 2, 'right_turn': 3}
# start positions
scenarios_start_pos = {'straight': 17, 'left_turn': 8, 'right_turn': 8}
