import numpy as np
import random
from neural_network.data_transformation import unit_test_sequences_data
import math


class AugmentData:
    def __init__(self, data, rotation_range=10, scaling_range=0.1, translation_range=0.1,
                 noise_level=0.01, temporal_perturbation_range=5, joint_dropout_prob=0.1,
                 speed_change_range=(0.8, 1.2), mirror_data=False, data_synthesis_method=None,
                 synthesis_ratio=0.5):
        """
        This class will be used to augment the data for the neural network
        :param data: (list) the data
        Should be in the form that the neural network accepts
        each trial will be a list of time steps inside each time step will be a list of 43 ints (right now may change) 0-41 is the xy of each point of the hand
        finally the last int will be the current state of the hand (hand in middle hand up or hand down)
        :param rotation_range: (int) of the degrees
        Description: This parameter controls the range of random rotation applied to the skeleton data.
        Use Case: You can specify a range (e.g., -10 to 10 degrees) to randomly rotate the entire skeleton data in 3D space to simulate different hand orientations and positions.
        :param scaling_range: (float)
        Description: This parameter controls the range of random scaling applied to the skeleton data.
        Use Case: You can specify a range (e.g., 0.1) to randomly scale the entire skeleton data, simulating variations in the size of the hand or distance from the camera.
        :param translation_range: (float)
        Description: This parameter controls the range of random translation applied to the skeleton data.
        Use Case: You can specify a range (e.g., 0.1) to randomly translate the entire skeleton data in 3D space, simulating variations in the hand's position within the frame.
        :param noise_level: (float) the percent (from 0-1) This will add noise to the data
        Description: This parameter controls the magnitude of random noise added to the skeleton data.
        Use Case: Adding noise to the joint positions helps the model become more robust to sensor inaccuracies or small variations in the data. The noise can be defined as a small percentage of the joint position values (e.g., 1% of the range of motion).
        :param temporal_perturbation_range:
        Description: This parameter controls the range for perturbing the timing and order of frames in sequences.
        Use Case: By specifying a range, you can randomly adjust the order and timing of frames in your sequence data. This simulates variations in the timing of hand movements.
        :param joint_dropout_prob:
        Description: This parameter defines the probability of dropping out (disabling) a joint in a frame.
        Use Case: Setting a dropout probability allows you to simulate situations where certain joints may not be reliably detected in the input data, thus improving the model's
        :param speed_change_range:
        Description: This parameter defines a range for modifying the speed of the movement sequences.
        Use Case: You can adjust the speed of your sequence data by specifying a range (e.g., 0.8 to 1.2) to simulate slower or faster hand movements. This helps the model generalize to different motion dynamics.
        :param mirror_data: (bool) if the data should be mirrored
        :param data_synthesis_method: (will not be used)
        :param synthesis_ratio: (will not be used)
        """
        self.origin_hand = data[0][42]
        # gets rid of the int original hand state part of each
        temp_data = []

        for frame in data:
            x = frame
            x.pop(42)
            temp_data.append(x)

        self.data = temp_data

        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.translation_range = translation_range
        self.noise_level = noise_level
        self.temporal_perturbation_range = temporal_perturbation_range
        self.joint_dropout_prob = joint_dropout_prob
        self.speed_change_range = speed_change_range
        self.mirror_data = mirror_data
        self.data_synthesis_method = data_synthesis_method
        self.synthesis_ratio = synthesis_ratio
        self.new_data = []

    def add_to_final_data(self, list_of_frames):
        """
        adds a new entry to the data
        :param list_of_frames: (list) the data
        :return: NA
        """
        return_list = []
        for frame in list_of_frames:
            temp = frame
            temp.append(self.origin_hand)
            return_list.append(temp)

        # for indiv in range(len(return_list)):
        #     for runner in range(len(return_list[indiv])):
        #         print(return_list[indiv][runner])
        #         return_list[indiv][runner] = math.floor(return_list[indiv][runner])

        # temporary fix i dont fucking know why this needs to be done
        for runner in range(len(return_list)):
            if len(return_list[runner]) == 44:
                return_list[runner].pop(43)

        try:

            unit_test_sequences_data(return_list)
        except Exception as e:
            pass
        self.new_data.append(return_list)



    def apply_random_rotation(self):
        """
        takes the data and rotates it adds 1 new data point to new_data
        :return:
        """
        min_angle = -self.rotation_range
        max_angle = self.rotation_range

        rotate_angle = np.radians(random.uniform(min_angle, max_angle))

        rotation_matrix = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)],
                                    [np.sin(rotate_angle), np.cos(rotate_angle)]])

        add_data = []

        for i in range(len(self.data)):
            temp_arr = []
            frame_array = np.array(self.data[i]).reshape((-1, 2))
            rotated_frame = np.rint(np.dot(frame_array, rotation_matrix)).astype(int)
            temp_arr.extend(rotated_frame.ravel().tolist())
            add_data.append(temp_arr)

        self.add_to_final_data(add_data)

    def apply_random_scaling(self):
        """
        takes the data and scales it adds 1 new data point to new_data
        :return:
        """
        min_scale = 1.0 - self.scaling_range
        max_scale = 1.0 + self.scaling_range

        scale_factor = random.uniform(min_scale, max_scale)

        add_data = []
        for i in range(len(self.data)):
            temp_adding = []
            for runner in range(len(self.data[i])):
                temp_adding.append(round(self.data[i][runner] * scale_factor))
            add_data.append(temp_adding)
        self.add_to_final_data(add_data)

    def apply_random_translation(self):
        min_translation = 1 - self.translation_range
        max_translation = 1 + self.translation_range

        translate = random.uniform(min_translation, max_translation)

        add_data = []
        for i in range(len(self.data)):
            new = []
            for indiv in range(len(self.data[i])):
                new_value = int(round(self.data[i][indiv] * translate))
                new.append(new_value)

            add_data.append(new)

        self.add_to_final_data(add_data)

    def apply_noise(self):
        noise_level = self.noise_level

        add_data = []
        for i in range(len(self.data)):
            frame = self.data[i]
            noisy_frame = []
            for coord in frame:
                noise = int(round(np.random.normal(0, noise_level)))
                noisy_coord = coord + noise
                noisy_frame.append(noisy_coord)
            add_data.append(noisy_frame)

        self.add_to_final_data(add_data)

    def apply_temporal_perturbation(self):
        # probably not going to be used
        # Define the temporal perturbation range, which determines how much the timing can change.
        min_perturbation = -self.temporal_perturbation_range
        max_perturbation = self.temporal_perturbation_range

        # Generate a random perturbation value for the timing of frames.
        perturbation = random.uniform(min_perturbation, max_perturbation)

        # Apply the perturbation to the order of frames.
        # changing self.data bad
        # if perturbation >= 0:
        #     self.data = self.data[int(perturbation):] + self.data[:int(perturbation)]
        # else:
        #     perturbation = abs(perturbation)
        #     self.data = self.data[-int(perturbation):] + self.data[:-int(perturbation)]

    def apply_joint_dropout(self):
        """
        randomly makes joints DISAPPEAR (magic)
        :return:
        """
        dropout_prob = self.joint_dropout_prob

        add_data = []
        for i in range(len(self.data)):

            frame = self.data[i]
            for j in range(1, len(frame), 2):
                if random.random() < dropout_prob:
                    frame[j] = 0
                    frame[j-1] = 0

            add_data.append(frame)

        self.add_to_final_data(add_data)

    def apply_speed_change(self):
        # not gonna use for now
        min_speed_change, max_speed_change = self.speed_change_range

        speed_change_factor = random.uniform(min_speed_change, max_speed_change)

        # Apply speed changes to the timing of frames.
        add_data = []
        for i in range(len(self.data)):
            frame = self.data[i]
            new_i = int(i * speed_change_factor)
            if new_i < len(self.data):
                add_data.append(self.data[new_i])

    def mirror_skeleton_data(self):
        # Create mirror-image data by flipping sequences horizontally.
        pass

    def augment_data(self):
        """
        will call all augmentation functions to augment the data Goal will be 500x data size
        :return:
        """
        total_data = 100
        total_runs = int(total_data/5)
        for x in range(total_runs):
            self.test_self_data()
            self.apply_random_rotation()
        for x in range(total_runs):
            self.test_self_data()
            self.apply_random_scaling()
        for x in range(total_runs):
            self.test_self_data()
            self.apply_random_translation()
        for x in range(total_runs):
            self.test_self_data()
            self.apply_noise()
        for x in range(total_runs):
            self.test_self_data()
            self.apply_joint_dropout()
        self.new_data.append(self.data)

    def test_self_data(self):
        for x in range(len(self.data)):
            if len(self.data[x]) == 44:
                self.data[x].pop(43)

if __name__ == '__main__':
    data = [[123, 256, 155, 312, 207, 349, 254, 364, 290, 379, 279, 299, 338, 310, 374, 314, 404, 317, 284, 265, 349, 270, 391, 273, 424, 275, 277, 233, 340, 232, 378, 235, 408, 239, 261, 202, 310, 193, 342, 192, 371, 194, 1], [123, 257, 156, 314, 208, 351, 255, 364, 290, 380, 280, 299, 339, 310, 375, 314, 406, 316, 285, 265, 349, 270, 391, 273, 425, 275, 277, 234, 340, 233, 378, 236, 410, 240, 260, 202, 309, 193, 341, 192, 371, 194, 1], [124, 253, 158, 309, 209, 347, 254, 361, 288, 377, 281, 295, 340, 306, 376, 310, 407, 313, 285, 261, 350, 268, 391, 270, 425, 272, 278, 230, 340, 228, 378, 231, 408, 234, 260, 199, 309, 189, 341, 188, 370, 191, 1], [124, 250, 157, 307, 208, 345, 254, 360, 289, 375, 282, 294, 340, 305, 376, 309, 407, 311, 286, 260, 350, 266, 391, 268, 424, 269, 280, 229, 341, 227, 378, 229, 408, 233, 263, 197, 311, 188, 342, 186, 371, 189, 1], [123, 250, 158, 309, 209, 346, 255, 360, 290, 375, 282, 295, 340, 305, 376, 309, 408, 311, 286, 260, 350, 266, 391, 268, 425, 269, 279, 229, 340, 227, 378, 230, 409, 234, 262, 197, 311, 188, 342, 187, 372, 189, 1], [122, 252, 155, 309, 207, 347, 254, 361, 289, 376, 280, 296, 339, 307, 376, 311, 407, 313, 283, 262, 349, 267, 391, 269, 425, 271, 275, 230, 339, 229, 378, 232, 409, 235, 258, 199, 308, 190, 341, 189, 370, 191, 1], [124, 253, 157, 309, 208, 346, 254, 361, 290, 375, 281, 295, 340, 306, 375, 309, 406, 312, 285, 260, 349, 265, 390, 268, 424, 269, 277, 229, 339, 227, 377, 229, 409, 232, 260, 197, 309, 188, 341, 186, 370, 188, 1], [125, 253, 158, 310, 209, 347, 255, 361, 290, 375, 281, 295, 340, 305, 376, 309, 407, 312, 284, 261, 349, 265, 390, 266, 424, 268, 277, 229, 338, 227, 377, 229, 408, 233, 259, 198, 308, 188, 340, 187, 370, 189, 1], [125, 253, 156, 311, 207, 348, 255, 361, 291, 376, 278, 296, 338, 306, 374, 310, 405, 313, 282, 261, 348, 266, 390, 268, 424, 270, 276, 230, 339, 228, 377, 230, 408, 234, 259, 199, 308, 190, 340, 188, 369, 190, 1], [124, 255, 156, 313, 210, 350, 256, 364, 292, 377, 280, 296, 338, 306, 374, 311, 405, 314, 284, 262, 349, 266, 390, 269, 424, 271, 277, 230, 339, 228, 377, 230, 407, 233, 260, 199, 309, 190, 340, 189, 369, 190, 1], [125, 255, 158, 313, 211, 350, 257, 364, 291, 378, 281, 297, 339, 307, 375, 310, 406, 312, 285, 263, 349, 267, 390, 269, 424, 271, 278, 231, 339, 229, 376, 230, 407, 234, 261, 199, 309, 190, 340, 189, 369, 191, 1], [123, 256, 155, 314, 206, 352, 253, 366, 289, 378, 279, 300, 337, 311, 373, 317, 404, 320, 283, 266, 348, 272, 389, 275, 422, 278, 276, 235, 339, 233, 376, 237, 406, 242, 260, 205, 308, 196, 341, 196, 370, 198, 1], [123, 252, 156, 307, 208, 344, 254, 359, 290, 373, 280, 296, 339, 304, 375, 307, 406, 310, 284, 262, 348, 265, 389, 267, 422, 269, 276, 231, 338, 227, 375, 229, 404, 233, 260, 199, 309, 190, 341, 188, 369, 190, 1], [124, 249, 155, 307, 208, 345, 255, 359, 291, 373, 281, 295, 340, 304, 376, 307, 406, 309, 285, 260, 349, 264, 390, 266, 424, 267, 278, 229, 339, 226, 376, 228, 407, 231, 261, 197, 309, 188, 340, 186, 369, 188, 1], [121, 256, 154, 313, 207, 348, 255, 360, 293, 373, 280, 297, 338, 309, 373, 314, 404, 318, 284, 264, 349, 270, 390, 274, 424, 277, 277, 234, 338, 233, 376, 237, 407, 242, 261, 205, 309, 197, 342, 197, 371, 201, 1], [123, 252, 157, 306, 211, 340, 259, 352, 296, 366, 281, 293, 341, 303, 376, 307, 407, 310, 285, 259, 350, 264, 391, 267, 424, 269, 278, 228, 340, 227, 378, 229, 408, 233, 261, 198, 310, 189, 342, 188, 371, 191, 1], [124, 250, 159, 307, 214, 339, 262, 349, 299, 362, 282, 289, 341, 298, 377, 300, 407, 302, 285, 255, 350, 258, 391, 260, 423, 262, 278, 224, 340, 221, 378, 222, 408, 226, 260, 193, 309, 183, 341, 182, 370, 184, 1], [120, 258, 154, 315, 210, 349, 259, 361, 296, 374, 280, 298, 339, 309, 375, 315, 407, 319, 284, 265, 349, 272, 391, 275, 425, 277, 277, 235, 339, 235, 378, 241, 408, 247, 262, 205, 311, 199, 343, 200, 372, 205, 1], [121, 258, 158, 314, 215, 345, 263, 355, 300, 368, 281, 298, 340, 307, 376, 310, 406, 312, 284, 264, 350, 268, 391, 270, 424, 272, 277, 233, 339, 231, 377, 233, 407, 237, 259, 203, 308, 195, 341, 194, 371, 196, 1], [120, 257, 156, 314, 213, 346, 262, 356, 299, 369, 281, 297, 340, 306, 376, 309, 407, 311, 284, 263, 350, 267, 392, 269, 425, 271, 276, 233, 339, 229, 378, 232, 408, 237, 258, 202, 308, 194, 341, 194, 371, 196, 1], [120, 256, 154, 313, 208, 347, 257, 358, 295, 371, 280, 296, 339, 309, 375, 314, 405, 317, 284, 263, 350, 270, 391, 273, 425, 275, 278, 233, 340, 232, 377, 235, 407, 239, 262, 203, 310, 195, 343, 195, 372, 198, 1]]

    partial_data = AugmentData(
        data=data,
        rotation_range=10,
        scaling_range=0.1,
        translation_range=0.1,
        noise_level=0.1,
        joint_dropout_prob=0.01,
    )

    # partial_data.augment_data()
    # for y in partial_data.new_data:
    #     print(y)
    # # print(partial_data.new_data)
    # # for y in partial_data.new_data:
    # #     print(y)
    # #     print(len(y))
    # #     print(len(y[0]))
