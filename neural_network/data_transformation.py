from neural_network.data_handling import run_txt_to_data,count_lines_in_file


def standardize_data(data):
    """
    Takes in data and makes it 21 frames
    This function makes my eyes bleed
    :param data: (list) of frames (lists) each frame has 43 (ints)
    :return:
    """
    if len(data) == 21:
        return data

    elif len(data) == 22:
        data.pop(19)
    elif len(data) == 23:
        data.pop(19)
        data.pop(9)
    elif len(data) == 24:
        data.pop(21)
        data.pop(14)
        data.pop(7)
    elif len(data) == 25:
        data.pop(20)
        data.pop(15)
        data.pop(10)
        data.pop(5)

    elif len(data) == 26:
        data.pop(25)
        data.pop(20)
        data.pop(15)
        data.pop(10)
        data.pop(5)
    elif len(data) == 20:
        data.insert(10, data[10])
    elif len(data) == 19:
        data.insert(10, data[10])
        data.insert(5, data[5])
    elif len(data) == 18:
        data.insert(10, data[10])
        data.insert(5, data[5])
        data.insert(2, data[2])
    elif len(data) == 17:
        data.insert(14, data[14])
        data.insert(10, data[10])
        data.insert(7, data[7])
        data.insert(5, data[5])
    elif len(data) == 16:
        data.insert(15, data[15])
        data.insert(12, data[12])
        data.insert(8, data[8])
        data.insert(5, data[5])
        data.insert(2, data[2])
    elif len(data) == 15:
        data.insert(14, data[14])
        data.insert(12, data[12])
        data.insert(10, data[10])
        data.insert(8, data[8])
        data.insert(5, data[5])
        data.insert(2, data[2])

    elif len(data) == 14:
        data.insert(13, data[13])
        data.insert(12, data[12])
        data.insert(10, data[10])
        data.insert(8, data[8])
        data.insert(5, data[5])
        data.insert(4, data[4])
        data.insert(2, data[2])

    elif len(data) == 13:
        data.insert(12, data[12])
        data.insert(10, data[10])
        data.insert(8, data[8])
        data.insert(6, data[6])
        data.insert(5, data[5])
        data.insert(3, data[3])
        data.insert(2, data[2])
        data.insert(1, data[1])
    else:
        return False
    return data


def unit_test_sequences_data(data):
    assert isinstance(data, list)
    assert len(data) == 21
    for frame in data:
        assert isinstance(frame, list)
        assert len(frame) == 43 or len(frame) == 42


def total_data_to_sequences(data):
    sequences = []
    for indiv in data:
        sequences.append(raw_data_to_sequences(indiv))
    return sequences


def raw_data_to_sequences(data):
    unit_test_prune_data(data)
    sequences = []
    for time_step in data[1]:
        sequence = hand_points_list_to_sequence(time_step)
        sequence.append(str_to_float_for_nn(data[0]))
        sequences.append(sequence)
    return sequences


def str_to_float_for_nn(inp):
    # 1 is hand is in middle 2 is hand is up and 3 is hand is down
    if inp == 'Start with HAND MIDDLE, ':
        ret = 1
    elif inp == 'Start with HAND UP, ':
        ret = 2
    else:
        # hand is down
        ret = 3
    return ret


def hand_points_list_to_sequence(hp_list: list) -> list:
    sequence = []
    for hand_point in hp_list:
        sequence.append(hand_point[1])
        sequence.append(hand_point[2])
    return sequence


def unit_test_prune_data(data):
    assert isinstance(data, list)
    assert isinstance(data[1], list)
    assert isinstance(data[0], str)
    for dat in data[1]:
        assert isinstance(dat, list)
        for da in dat:
            assert is_list_of_3_ints(da)


def is_list_of_3_ints(lst):
    return isinstance(lst, list) and len(lst) == 3 and all(isinstance(item, int) for item in lst)


def test_data_structure(data):
    assert isinstance(data, list), "Data is not a list"
    assert len(data) in (12, 13), "Data should contain 12 or 13 inner lists"

    for inner_list in data:
        assert isinstance(inner_list, list), "Inner list is not a list"
        assert len(inner_list) == 21, "Inner list should contain 21 innermost lists"
        for innermost_list in inner_list:
            assert is_list_of_3_ints(innermost_list), "Innermost list should contain 3 integers"


def data_file_to_code(data_file):
    return_data = []
    for x in range(count_lines_in_file(filename=data_file)):
        data = run_txt_to_data(data_file, x + 1)
        return_data.append(data)

    return return_data


def test_neural_network_data(data):
    assert isinstance(data, list), "Data is not a list"
    assert len(data) in (12, 13), "Data should contain 12 or 13 inner lists"
    for innerlist in data:
        assert isinstance(innerlist, list), "Inner list is not a list"
        assert len(innerlist) == 42, "Inner list should contain 21 innermost lists"
        for inner_data in innerlist:
            assert isinstance(inner_data, int), "Inner list is not a int"


def training_data_to_neural_network_ready_data(data):
    """
    will take 1 list in data
    This function will change this is iteration 1
    will need a new name after this training fails and training data changes
    :return:
    """


    new_data = []
    for time_step in data:
        new_time_step = []
        for hand_point in time_step:
            new_time_step.append(hand_point[1])
            new_time_step.append(hand_point[2])
        new_data.append(new_time_step)
    # bug fix some data didnt have the same lenght for unknown reasons
    if len(new_data) == 13:
        new_data.append(new_data[12])
    if len(new_data) == 12:
        new_data.append(new_data[11])
        new_data.append(new_data[11])
    if len(new_data) == 11:
        new_data.append(new_data[10])
        new_data.append(new_data[10])
        new_data.append(new_data[10])
    if len(new_data) == 15:
        new_data.pop(14)

    return new_data


def testing_output_from_training_data_to_neural_network_ready_data(data):
    """
    Only reason for this function is some data from training_data_to_neural_network_ready_data has no length gonna weed them out
    :param data:
    :return:
    """
    if len(data[0]) == 0:
        return False

    runner = 0
    total_zeros = 0
    while runner < len(data):
        if len(data[runner]) == 0:
            total_zeros += 1
            data[runner] = data[runner-1]
        runner += 1
    if total_zeros > 5:
        return False
    else:
        return data


def labels_to_final_label_ready_for_neural_network(labels):
    new_labels = []
    for label in labels:
        if label == 'HAND DOWN':
            new_labels.append([1.0, 0, 0])
        elif label == 'HAND UP':
            new_labels.append([0, 1.0, 0])
        else:
            new_labels.append([0, 0, 1.0])
    return new_labels


def input_from_script_to_nn_ready(frame_accumulator, current_position=1):
    sequences = []
    for frame in frame_accumulator:
        individual_sequence = []
        for individual in frame:
            individual_sequence.append(individual[1])
            individual_sequence.append(individual[2])
        individual_sequence.append(current_position)
        sequences.append(individual_sequence)
    unit_test_sequences_data(sequences)
    return sequences