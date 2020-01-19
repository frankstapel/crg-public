import time
from math import sqrt, atan, pi, e, sin, cos, ceil
from os import path, stat
import gspread
import pickle
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np


def main():
    # Used to measure the runtime of the algorithm
    start_time = time.time()

    board = get_board()
    # [grade, [move1, ..], [hold1, ..]]
    # Grade can be 6A, 6B, 6C, 7A, 7B, 7C, 8A
    # Hold can be Jug, Edge, Sloper, Pinch, Pocket
    # Length adjuster can be -1, 0 (standard), 1 or 2
    route_requirements = ['7A', ['Jug'], 0]
    if route_requirements is None:
        route_requirements = get_route_requirements()

    # Generate a single route
    # route = generate_route(board, route_requirements)
    # print(get_board_string(board, route))

    # Generate a tree and print the routes
    tree_routes = generate_route_tree(board, route_requirements)
    number_of_printed_routes = 10
    if len(tree_routes) >= number_of_printed_routes:
        for i in range(number_of_printed_routes):
            print(get_board_string(board, tree_routes[i][0]))
            print('\nRoute ' + str(i + 1) + ' has a final score of ' + str(tree_routes[i][1]) + '.\n\n')

    # Some runtime info
    duration = time.time() - start_time
    if duration < 60:
        print('\nThis took only ' + str(time.time() - start_time) + ' seconds!')
    elif duration < 3600:
        print('\nThis took ' + str(int((time.time() - start_time) / 60)) + ' minutes and ' + str(int(
            (time.time() - start_time) % 60)) + ' seconds!')


def generate_route(board, route_requirements):
    grade = string_to_grade(route_requirements[0])
    holds = route_requirements[1]
    distance_modifier = route_requirements[2]

    # [[column, row, type], ...]
    # Type can be: s (start), m (move), f (finish)
    route = []
    number_of_holds = 0

    # Used for determining the position of the next hold
    current_hand = None
    old_hand = None
    finish_hold_found = False

    # Add starting holds
    while not finish_hold_found:
        scores = get_scores(board, route, grade, holds, number_of_holds, current_hand, old_hand, distance_modifier)

        # Find best scoring hold
        max_score = 0.0
        max_row = 0
        max_col = 0
        for col_index, column in enumerate(scores):
            for row_index, score in enumerate(column):
                if score[0] > max_score:
                    max_score = score[0]
                    max_col = col_index
                    max_row = row_index

        # Update current holds
        old_hand = current_hand
        current_hand = [max_col, max_row]

        # Add hold to route with correct label
        hold_label = get_hold_label(max_col, max_row, number_of_holds)
        route.append(hold_label)
        number_of_holds += 1

        # Check if the route has ended
        if max_row == 17:
            finish_hold_found = True
    return route


def generate_route_tree(board, route_requirements):
    grade = string_to_grade(route_requirements[0])
    holds = route_requirements[1]
    distance_modifier = route_requirements[2]

    # [8, 12] - This combination gives good results in about 40 seconds
    leafs_per_node = 5  # 8, 5, 4, 3, 2, 2, 2, etc...
    max_iterations = 12  # 12 includes most routes

    current_next_nodes = [Node()]
    final_nodes = []
    final_routes = []
    iterations = 0

    while current_next_nodes and iterations <= max_iterations:
        print('Current length of route is ' + str(iterations))
        iterations += 1
        new_next_nodes = []
        for node in current_next_nodes:
            scores = get_scores(board, node.get_current_route(), grade, holds, node.get_number_of_holds(),
                                node.get_current_hand(), node.get_old_hand(), distance_modifier)
            all_scores = []
            for col_index, column in enumerate(scores):
                for row_index, score in enumerate(column):
                    all_scores.append([scores[col_index][row_index][0], col_index, row_index])
            all_scores = sorted(all_scores, key=lambda x: x[0], reverse=True)
            for i in range(leafs_per_node):
                score, column, row = all_scores[i]

                # print('Hold score = ' + str(scores[column][row][1]))
                # print('Difficulty score = ' + str(scores[column][row][2]))
                # print('Move score = ' + str(scores[column][row][3]))
                # print('Rotation score = ' + str(scores[column][row][4]) + '\n')

                # The next lines make sure a lot of suboptimal routes don't get generated.
                if score < 0.50:
                    break
                new_route = node.get_current_route() + [get_hold_label(column, row, node.get_number_of_holds())]
                finished = row == 17
                new_node = Node(node.get_number_of_holds() + 1, new_route, [column, row], node.get_current_hand(),
                                finished, node.get_scores() + [score])
                node.add_child(new_node)
                if finished:
                    final_nodes.append(new_node)
                    average_route_score = np.average(new_node.get_scores())
                    route_score = get_route_score(board, new_node.get_current_route(), grade, holds, distance_modifier)
                    final_route_score = (average_route_score + route_score) / 2.0
                    final_routes.append([new_node.get_current_route(), final_route_score])
                else:
                    new_next_nodes.append(new_node)
        leafs_per_node = get_new_leafs_per_node(leafs_per_node)
        current_next_nodes = new_next_nodes

    final_routes = sorted(final_routes, key=lambda x: x[1], reverse=True)

    print('Generated ' + str(len(final_routes)) + ' routes!')
    return final_routes


def get_scores(board, route, grade, holds, number_of_holds, current_hand, old_hand, distance_modifier):
    scores = get_initial_scores(route)

    if number_of_holds >= 2:
        position = get_position(old_hand, current_hand)
        goal_positions = get_goal_positions(board, grade, old_hand, current_hand, position, number_of_holds, distance_modifier)
    else:
        position = None
        goal_positions = None

    # Iterate through all the holds, determining a score for each hold
    for col_index in range(11):
        for row_index in range(18):
            if hold_in_route([col_index, row_index], route):
                continue

            # Check how good the hold type fits the route
            hold_score = get_hold_score(holds, board, col_index, row_index)

            # Check how good the difficulty fits the route
            difficulty_score = get_difficulty_score(grade, board, col_index, row_index)

            # Check how good the position matches the route (and difficulty!)
            move_score = get_move_score(board, current_hand, number_of_holds, goal_positions, col_index, row_index)

            # Check if the orientation of the hold matches the current position
            rotation_score = get_rotation_score(board, number_of_holds, position, goal_positions, col_index, row_index)

            # Combine found scores for final score
            scores[col_index][row_index] = get_total_score(scores[col_index][row_index][0], hold_score,
                                                           difficulty_score, move_score, rotation_score)
    return scores


class Node(object):
    def __init__(self, number_of_holds=0, current_route=None, current_hand=None, old_hand=None, finished=False,
                 scores=None):
        if current_route is None:
            current_route = []
        self.number_of_holds = number_of_holds
        self.current_route = current_route
        self.current_hand = current_hand
        self.old_hand = old_hand
        self.finished = finished
        self.scores = scores
        if scores is None:
            scores = []
        self.scores = scores
        self.children = []
        if self.number_of_holds == 2 and self.current_hand[1] < self.old_hand[1]:
            self.current_hand, self.old_hand = self.old_hand, self.current_hand

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_children(self):
        return self.children

    def get_current_route(self):
        return self.current_route

    def get_number_of_holds(self):
        return self.number_of_holds

    def get_current_hand(self):
        return self.current_hand

    def get_old_hand(self):
        return self.old_hand

    def is_finished(self):
        return self.finished

    def get_scores(self):
        return self.scores


def get_route_score(board, route, grade, holds, distance_modifier):
    distance_score, distance_deviation_score = get_distance_score(route, grade, distance_modifier)
    grade_score, grade_deviation_score = get_grade_score(board, route, grade)
    # centered_score = get_centered_score(route)
    hold_type_score = get_hold_type_score(board, route, holds)
    number_of_holds_score = get_number_of_holds_score(route, grade)
    weights = [2, 1, 4, 2, 2, 1]
    total_weight = sum(weights)
    return distance_score * weights[0] / total_weight + \
        distance_deviation_score * weights[1] / total_weight + \
        grade_score * weights[2] / total_weight + \
        grade_deviation_score * weights[3] / total_weight + \
        hold_type_score * weights[4] / total_weight + \
        number_of_holds_score * weights[5] / total_weight


def get_distance_score(route, grade, distance_modifier):
    distances = []
    for i in range(len(route) - 1):
        distances.append(get_distance(route[i], route[i + 1]))
    distance_average = np.average(distances)
    average_deviation = abs(distance_average - (0.5 * grade + 2.5 + distance_modifier))
    distance_score = e ** (-1.0 * (average_deviation / 3.0) ** 2)
    distance_deviation = np.std(distances)
    deviation_score = e ** (-1.0 * distance_deviation ** 2)
    if distance_score > 0.8:
        distance_score = 1.0
    if deviation_score > 0.8:
        deviation_score = 1.0
    return distance_score, deviation_score


def get_grade_score(board, route, grade):
    difficulties = []
    for hold in route:
        difficulties.append(board[hold[0]][hold[1]][0])
    difficulty_average = np.average(difficulties)
    average_deviation = abs(difficulty_average - grade / 0.6)
    difficulty_score = e ** (-1.0 * (average_deviation / 2.0) ** 2)
    difficulty_deviation = np.std(difficulties)
    deviation_score = e ** (-1.0 * difficulty_deviation ** 2)
    return difficulty_score, deviation_score


def get_centered_score(route):
    distances = []
    for hold in route:
        distances.append(hold[0])
    average_distance = abs(5 - np.average(distances))
    return e ** (-1 * (average_distance / 3) ** 2)


def get_hold_type_score(board, route, holds):
    major_hold_score = 0.0
    for hold in route:
        if board[hold[0]][hold[1]][2] in holds:
            major_hold_score += 1.0
    major_hold_score /= len(route)
    minor_hold_score = 0.0
    for hold in route:
        for hold_type in board[hold[0]][hold[1]][3]:
            if hold_type in holds:
                minor_hold_score += 1.0
    minor_hold_score /= 2 * len(route)
    return 0.8 * major_hold_score + 0.2 * minor_hold_score


def get_number_of_holds_score(route, grade):
    return e ** (-1 * (abs(-0.5 * grade + 10.0 - len(route)) / 3.0) ** 2)


def get_total_score(current_score, hold_score, difficulty_score, move_score, rotation_score):
    current_weight = 1.0
    hold_weight = 1.0
    difficulty_weight = 2.0
    move_weight = 3.0
    rotation_weight = 3.0
    total_weight = current_weight + hold_weight + difficulty_weight + move_weight + rotation_weight
    score = current_score * current_weight / total_weight + \
        hold_score * hold_weight / total_weight + \
        difficulty_score * difficulty_weight / total_weight + \
        move_score * move_weight / total_weight + \
        rotation_score * rotation_weight / total_weight
    return [score, hold_score, difficulty_score, move_score, rotation_score]


def get_initial_scores(route):
    scores = [[[1.0, 0.0, 0.0, 0.0] for _ in range(18)] for _ in range(11)]
    for hold in route:
        scores[hold[0]][hold[1]][0] = 0.0
    return scores


def get_hold_score(holds, board, col_index, row_index):
    if not holds:
        hold_score = 1.0  # type: float
    else:
        major_matches = 0.0
        minor_matches = 0.0
        for hold in holds:
            if hold == board[col_index][row_index][2]:
                major_matches += 1.0
            if hold in board[col_index][row_index][3]:
                minor_matches += 1.0
        hold_score = major_matches * 0.7 + minor_matches * 0.15
        number_of_holds = len(holds)
        if number_of_holds <= 1:
            hold_score = hold_score / 0.6
        elif number_of_holds <= 2:
            hold_score = hold_score / 0.8
        if hold_score < 0.1:
            hold_score = 0.1
    return hold_score


def get_difficulty_score(grade, board, col_index, row_index):
    return e ** (-1 * (abs(board[col_index][row_index][0] - grade / 0.6) / 3.0) ** 2)
    # difficulty_score = -0.025 * (board[col_index][row_index][0] - grade / 0.6) ** 2 + 1.0
    # if difficulty_score < 0.1:
    #     difficulty_score = 0.1
    # return difficulty_score


def get_move_score(board, current_hand, number_of_holds, goal_positions, col_index, row_index):
    move_score = 0.0
    if number_of_holds == 0:
        # Make sure first hold is pointing upwards
        if 1 < row_index < 6 and is_rotation_match(board, col_index, row_index, [0, 1, 7]):
            move_score = 1.0
        elif row_index <= 1:
            move_score = 0.1
    elif number_of_holds == 1:
        if 1 < row_index < 6 and get_distance(current_hand, [col_index, row_index]) < 5.0:
            if col_index < current_hand[0] and is_rotation_match(board, col_index, row_index, [0, 6, 7]):
                move_score = 1.0
            elif col_index > current_hand[0] and is_rotation_match(board, col_index, row_index, [0, 1, 2]):
                move_score = 1.0
            elif col_index == current_hand[0] and is_rotation_match(board, col_index, row_index, [0, 1, 7]):
                move_score = 1.0
            else:
                move_score = 0.1
        elif row_index <= 1 and get_distance(current_hand, [col_index, row_index]) < 5.0:
            move_score = 0.1
    else:
        max_score = 0.0
        for goal_position in goal_positions:
            goal_position_score = e ** (-1 * (get_distance(goal_position, [col_index, row_index]) / 3) ** 2)
            if goal_position_score > max_score:
                max_score = goal_position_score
        move_score = max_score
        if move_score > 0.8:
            move_score = 1.0
    return move_score


def is_rotation_match(board, col_index, row_index, allowed_rotations):
    rotation_match = False
    for i in allowed_rotations:
        if i in board[col_index][row_index][1]:
            rotation_match = True
    return rotation_match


def get_new_leafs_per_node(leafs_per_node, slow=False):
    if slow:
        if leafs_per_node > 3:
            return leafs_per_node - 1
        else:
            return 3
    else:
        return ceil(2.0 * leafs_per_node / 3.0)
        # return ceil(leafs_per_node / 2.0)
    # if leafs_per_node >= 8:
    #     return 5
    # elif leafs_per_node >= 5:
    #     return 4
    # elif leafs_per_node >= 4:
    #     return 3
    # else:
    #     return 2


def get_rotation_score(board, number_of_holds, position, goal_positions, col_index, row_index):
    if number_of_holds < 2:
        return 1.0
    else:
        goal_direction = get_position_direction(position, goal_positions[2])
        min_difference = 4.0
        difference = 4.0
        for hold_direction in board[col_index][row_index][1]:
            difference = abs(goal_direction - hold_direction)
            if difference < min_difference:
                min_difference = difference
        return e ** (-1.0 * (difference / 3.0) ** 2)


def get_goal_positions(board, grade, old_hand, current_hand, position, number_of_holds, distance_modifier):
    # Returns goal positions based on the current position.
    goal_positions = []
    if number_of_holds >= 2:
        distance = 0.5 * grade + 2.5 + distance_modifier
        for i in range(-2, 3):
            position_score, position_direction = get_position_score(board, old_hand, current_hand)
            radian_direction = position_direction * pi / 4.0 + i / 2.0
            goal_positions.append(
                [position[0] + distance * sin(radian_direction), position[1] + distance * cos(radian_direction)])
        if old_hand[1] <= current_hand[1]:
            top_hold = current_hand
            bottom_hold = old_hand
        else:
            top_hold = old_hand
            bottom_hold = current_hand
        goal_positions.append([top_hold[0], top_hold[1] + distance])
        if top_hold[0] != bottom_hold[0] and top_hold[1] != bottom_hold[1]:
            if top_hold[0] > bottom_hold[0]:
                dx = 0.3 * grade + 0.5
            else:
                dx = -(0.3 * grade + 0.5)
            goal_positions.append([top_hold[0] + dx, top_hold[1] + 0.4 * grade + 2.0])
    else:
        goal_positions.append([0.0, 0.0])
    return goal_positions


def get_hold_label(max_col, max_row, number_of_holds):
    # Returns the correctly labeled hold
    if max_row < 6 and number_of_holds < 2:
        return [max_col, max_row, 's']
    elif max_row == 17:
        return [max_col, max_row, 'f']
    else:
        return [max_col, max_row, 'm']


def get_distance(h1, h2):
    # Returns the distance between two holds
    if h1 is not None and h2 is not None:
        return sqrt((h1[0] - h2[0]) ** 2 + (h1[1] - h2[1]) ** 2)
    else:
        return 0


def get_position(old_hand, current_hand):
    # Returns the center position of the two current holds
    return [(old_hand[0] + current_hand[0]) / 2, (old_hand[1] + current_hand[1]) / 2]


def get_position_score(board, old_hand, current_hand):
    # Returns a score for the current position
    same_width = False
    if old_hand[0] < current_hand[0]:
        lefts = board[old_hand[0]][old_hand[1]][1]
        left_hold = old_hand
        rights = board[current_hand[0]][current_hand[1]][1]
        right_hold = current_hand
    elif old_hand[0] > current_hand[0]:
        lefts = board[current_hand[0]][current_hand[1]][1]
        left_hold = current_hand
        rights = board[old_hand[0]][old_hand[1]][1]
        right_hold = old_hand
    else:
        lefts = board[old_hand[0]][old_hand[1]][1]
        left_hold = old_hand
        rights = board[current_hand[0]][current_hand[1]][1]
        right_hold = current_hand
        same_width = True

    best_position = 0.0
    best_direction = 0.0

    for left in lefts:
        for right in rights:
            # average hold_direction
            hold_direction = ((left + right) % 8.0) / 2.0
            if 2.0 < hold_direction < 6.0:
                hold_direction = (hold_direction + 4.0) % 8.0

            # position direction
            position_direction = get_position_direction(left_hold, right_hold)

            # score for that hold_direction
            hold_relation = e ** (-1.0 / 8.0 * (right - left - 1.0) ** 2)

            # average of the two directions
            average_direction = (hold_direction + position_direction % 8.0) % 8.0 / 2.0
            if same_width:
                average_direction = hold_direction

            # deviation of the two directions
            direction_deviation = -0.1 * abs(position_direction % 8 - hold_direction) ** 2 + 1

            # The score of a hold based on the hold_relation and direction deviation
            score = hold_relation * direction_deviation

            # Update best values
            if score > best_position:
                best_position = score
                best_direction = average_direction
    return [best_position, best_direction]


def get_position_direction(p1, p2):
    # Gets the direction based on the current two holds, returns 0 if holds are placed above each other
    if p2[0] - p1[0] == 0:
        return 0.0
    else:
        return ((atan(abs(p2[1] - p1[1]) / (p2[0] - p1[0])) * 4.0) / pi) % 8.0


def hold_in_route(hold, route):
    # Returns if a hold is already in a route
    for route_hold in route:
        if hold[0] == route_hold[0] and hold[1] == route_hold[1]:
            return True
    return False


def grade_to_string(grade):
    # Gets the grade of a number representation
    strings = {
        0: '6A',
        1: '6B',
        2: '6C',
        3: '7A',
        4: '7B',
        5: '7C',
        6: '8A'
    }
    return strings[grade]


def string_to_grade(string):
    # Converts grades to their number representation
    grades = {
        '6A': 0,
        '6B': 1,
        '6C': 2,
        '7A': 3,
        '7B': 4,
        '7C': 5,
        '8A': 6
    }
    return grades[string]


def get_board_string(board, route):
    # Prints a route on a board to a string
    start_color = '\033[92m'
    finish_color = '\033[91m'
    move_color = '\033[94m'
    end_color = '\033[0m'
    result = '\n   A B C D E F G H I J K\n'
    for row in range(17, -1, -1):
        if row > 8:
            line = str(row + 1)
        else:
            line = ' ' + str(row + 1)
        for column in range(11):
            hold_found = False
            for hold in route:
                if hold[0] == column and hold[1] == row and not hold_found:
                    line += ' '
                    # print the rounded difficulty
                    difficulty = str(int(board[column][row][0]))
                    if hold[2] == 's':
                        line += start_color + difficulty + end_color
                    elif hold[2] == 'f':
                        line += finish_color + difficulty + end_color
                    else:
                        line += move_color + difficulty + end_color
                    hold_found = True
            if not hold_found:
                line += ' .'
        line += ' '
        line += str(row + 1)
        line += '\n'
        result += line
    result += '   A B C D E F G H I J K\n'
    return result


def get_board():
    # Gets a board either from an existing file or from
    # to force refresh the board data, delete the file ../misc/board.pkl
    if not path.exists('../misc/board.pkl'):
        open('../misc/board.pkl', 'a').close()
    if stat('../misc/board.pkl').st_size != 0:
        with open('../misc/board.pkl', 'rb') as board_input:
            board = pickle.load(board_input)
    else:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            '../misc/Climbing Route Generation-f8a5fdbc71a5.json', scope)
        gc = gspread.authorize(credentials)
        classification = gc.open_by_key('1pwnKZJPTzYeM6ImzxG5_9y8vLy969lIaJesnB_LVEQ0').worksheet(
            'Final Classification')
        records = classification.get_all_values()

        # Setup a board of 11 wide and 18 high
        board = [[[] for _ in range(18)] for _ in range(11)]

        # Go over each row of the MoonBoard
        for column in range(11):  # type: int
            # Go over each column of the MoonBoard
            for row in range(18):  # type: int
                data = records[(18 - row) + 18 * column]
                minor_hold_types = []
                if data[5] != '':
                    minor_hold_types.append(data[5])
                if data[6] != '':
                    minor_hold_types.append(data[6])
                # Add the data to the board
                # [difficulty, set of rotations, major hold type, set of minor hold types]
                board[column][row] = [float(data[2].replace(",", ".")), set([int(x) for x in data[3].split('$')]),
                                      data[4],
                                      set(minor_hold_types)]

        with open('../misc/board.pkl', 'wb') as board_output:
            pickle.dump(board, board_output, pickle.HIGHEST_PROTOCOL)
    return board


def get_route_requirements():
    # Textual interface for the route requirements
    print('Please enter desired route requirements.')
    grade = input('Grade: ')
    grades = {'6A', '6B', '6C', '7A', '7B', '7C', '8A'}
    if grade not in grades:
        grade = '7A'
        print('Grade not recognized, used default grade 7A.')
    # moves = input('Moves, separated by commas: ').split(', ')
    holds = input('Holds, separated by commas. Leave empty to use all available holds: ').split(', ')
    distance_modifier = input('Distance modifier. Can be -1, 0, 1 or 2. For standard enter 0: ')
    return [grade, holds, distance_modifier]


if __name__ == '__main__':
    main()
