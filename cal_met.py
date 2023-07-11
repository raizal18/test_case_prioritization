import  numpy as np

def calculate_apfd(actual_order, predicted_order):
    total_faults = len(actual_order)
    total_steps = total_faults * 2
    total_positions = 0
    sum_positions = 0
    
    for i, test_case in enumerate(actual_order):
        if test_case in predicted_order:
            position = predicted_order.index(test_case) + 1
            total_positions += 1
            sum_positions += position
    
    apfd = 1 - (sum_positions / total_positions) / total_steps + 1 / (2 * total_faults)
    return apfd

def calculate_apfdc(apfd, total_cost):
    apfdc = apfd / total_cost
    return apfdc

def calculate_napfd(apfd, total_faults):
    napfd = apfd / (1 - 1 / (2 * total_faults))
    return napfd

def calculate_apfd_ta(actual_order, predicted_order, execution_times):
    total_faults = len(actual_order)
    total_steps = total_faults * 2
    total_positions = 0
    sum_time_aware_positions = 0
    
    for i, test_case in enumerate(actual_order):
        if test_case in predicted_order:
            position = predicted_order.index(test_case) + 1
            time_aware_position = position / execution_times[test_case]
            total_positions += 1
            sum_time_aware_positions += time_aware_position
    
    apfd_ta = (sum_time_aware_positions / total_positions) / total_steps
    return apfd_ta

def calculate_rmse(predictions, ground_truth):
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    return rmse

# Assuming you have the following variables:
# actual_order: List containing the actual order of faults
# predicted_order: List containing the predicted order of faults
# execution_times: Dictionary containing execution times for each test case
# total_cost: Total cost of executing all test cases
# predictions: Numpy array containing model predictions
# ground_truth: Numpy array containing ground truth values

# # Calculate APFD
# apfd = calculate_apfd(actual_order, predicted_order)

# # Calculate APFDc
# apfdc = calculate_apfdc(apfd, total_cost)

# # Calculate NAPFD
# total_faults = len(actual_order)
# napfd = calculate_napfd(apfd, total_faults)

# # Calculate APFD from APFDc
# apfd_from_apfdc = apfdc * total_cost

# # Calculate APFD_TA
# apfd_ta = calculate_apfd_ta(actual_order, predicted_order, execution_times)

# # Calculate RMSE
# rmse = calculate_rmse(predictions, ground_truth)

# # Print or use the calculated metrics as needed
# print("APFD:", apfd)
# print("APFDc:", apfdc)
# print("NAPFD:", napfd)
# print("APFD from APFDc:", apfd_from_apfdc)
# print("APFD_TA:", apfd_ta)
# print("RMSE:", rmse)
