
def filter_tuples(input_list):
    results = []
    series = []
    encountered_1 = False

    for tup in input_list:
        _, b = tup

        if not series:
            # If series is empty, add the tuple with b == 1 to the series
            if b == 1:
                series.append(tup)
                encountered_1 = True
            continue

        # Check if the current tuple breaks the series
        if (b == 1 and series[-1][1] == -1) or (b == -1 and series[-1][1] == 1):
            if not encountered_1:
                # If we haven't encountered 1 yet, omit the series
                series = []
            else:
                # Otherwise, apply the rules and add to results
                if len(series) > 4 and series[0][1] == 1:
                    results.extend(series[:2] + series[-2:])
                else:
                    results.extend(series)
                series = []

        series.append(tup)

    # Handle the last series
    if series:
        if len(series) > 4 and series[0][1] == 1:
            results.extend(series[:2] + series[-2:])
        else:
            results.extend(series)

    # Handle the case where the list starts with a series of b == 1 
    initial_ones_count = 0 
    for tup in input_list:
        _, b = tup
        if b == 1:
            initial_ones_count += 1
        else: break
    if initial_ones_count > 2:
        results = results[initial_ones_count - 2:]

    return results

# # Test cases
# input1 = [(0.022, 1), (0.023, 1), (0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1)]
# output1 = filter_tuples(input1)
# assert(output1 == [ (0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1)])

# input2 = [(0.022, -1), (0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1)]
# output2 = filter_tuples(input2)
# assert(output2 == [ (0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1)])

# input3 = [(0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1),
#           (1.487, 1), (1.497, 1), (1.513, 1), (1.707, 1), (1.71, -1), (1.883, -1), (1.89, 1), (2.403, 1),
#           (2.405, -1), (2.583, -1), (2.585, 1), (2.962, 1), (2.988, 1), (3.002, 1), (4.81, 1), (5.045, 1),
#           (5.047, -1), (5.228, -1), (5.23, 1), (5.477, 1), (5.478, -1), (5.657, -1), (5.658, 1), (5.952, 1),
#           (6.028, -1), (6.157, -1), (6.442, 1), (6.455, 1), (8.853, 1), (9.06, 1), (9.067, -1), (9.252, -1),
#           (9.26, 1), (9.54, 1)]
# output3 = filter_tuples(input3)
# assert(output3 == [(0.73, 1), (0.908, 1), (0.917, -1), (1.085, -1), (1.1, 1), (1.313, 1),
#           (1.513, 1), (1.707, 1), (1.71, -1), (1.883, -1), (1.89, 1), (2.403, 1),
#           (2.405, -1), (2.583, -1), (2.585, 1), (2.962, 1), (4.81, 1), (5.045, 1),
#           (5.047, -1), (5.228, -1), (5.23, 1), (5.477, 1), (5.478, -1), (5.657, -1), (5.658, 1), (5.952, 1),
#           (6.028, -1), (6.157, -1), (6.442, 1), (6.455, 1), (8.853, 1), (9.06, 1), (9.067, -1), (9.252, -1),
#           (9.26, 1), (9.54, 1)])