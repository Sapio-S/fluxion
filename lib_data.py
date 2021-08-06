import csv, math, os, random, statistics, sys

def readCSVFile(inputFileNames:list, sample_x_names:list, samples_y_names:list, samples_y_weights:list):
    samples_x = []
    samples_y = []
    samples_y_aggregation = []
    err_msg = None
    
    # Sanity check on function arguments
    if type(sample_x_names) != list or len(sample_x_names) == 0: 
        err_msg = "\"sample_x_names\" is not a list"
    elif type(samples_y_names) != list or len(samples_y_names) == 0:
        err_msg = "\"samples_y_names\" is not a list"
    elif type(samples_y_weights) != list:
        err_msg = "\"samples_y_weights\" is not a list"
    elif len(samples_y_weights) != len(samples_y_names):
        err_msg = "The size of \"samples_y_weights\" and \"samples_y_names\" does not match"
    
    if err_msg is None:
        try:
            for inputFileName in inputFileNames:
                with open(inputFileName, "r") as csvfile:
                    for row in csv.DictReader((row for row in csvfile if not row.startswith("#")), delimiter=",", skipinitialspace=True):
                        x = []
                        for samples_x_name in sample_x_names:
                            x.append(float(row[samples_x_name]))
                        
                        samples_y_value = 0
                        for idx in range(len(samples_y_names)):
                            samples_y_value += float(row[samples_y_names[idx]]) * samples_y_weights[idx]
                        
                        try:
                            idx = samples_x.index(x)  # Checks whether we have x already
                            samples_y[idx].append(samples_y_value)
                        except ValueError:
                            samples_x.append(x)
                            samples_y.append([samples_y_value])
            
            # Aggregate multiple observation of the sample sampling points
            samples_y_aggregation = [statistics.median(sample_y) for sample_y in samples_y]
        except Exception as e:
            if type(e) == KeyError:
                err_msg = "Key is not found: %s" % (str(e))
            elif type(e) == ValueError:
                err_msg = "ValueError: %s" %  (str(e))
            else:
                err_msg = "Unexpected error occurred"
    
    if err_msg != None:
        sys.stderr.write("[%s] Errors occurred while processing CSV file: %s\n" % (os.path.basename(__file__), err_msg))
        samples_x.clear()
        samples_y.clear()
        samples_y_aggregation.clear()
    
    return samples_x, samples_y, samples_y_aggregation, err_msg

# Update values in the array, to match their corresponding type
def match_val_type(vals, vals_bounds, vals_types):
    vals_new = []
    
    for i in range(len(vals_types)):
        if vals_types[i] == "discrete_int":
            # Find the closest integer in the array, vals_bounds
            vals_new.append(min(vals_bounds[i], key=lambda x: abs(x - vals[i])))
        elif vals_types[i] == "range_int":
            # Round down to the nearest integer
            vals_new.append(math.floor(vals[i]))
        elif vals_types[i] == "range_continuous":
            # Don't do any processing for continous numbers
            vals_new.append(vals[i])
        else:
            return None
    
    return vals_new

def rand(x_bounds, x_types):
    outputs = []

    for i in range(0, len(x_bounds)):
        if x_types[i] == "discrete_int":
            temp = x_bounds[i][random.randint(0, len(x_bounds[i]) - 1)]
            outputs.append(temp)
        elif x_types[i] == "range_int":
            temp = random.randint(x_bounds[i][0], x_bounds[i][1])
            outputs.append(temp)
        elif x_types[i] == "range_continuous":
            temp = random.uniform(x_bounds[i][0], x_bounds[i][1])
            outputs.append(temp)
        else:
            return None
    
    return outputs
    