import csv, math, os, random, statistics, sys

def readCSVFile(inputFileNames:list, sample_x_names:list, samples_y_name:str, samples_y_aggregator=statistics.median):
    samples_x = []
    samples_y = []
    samples_y_aggregation = []
    err_msg = None
    
    # Sanity check on function arguments
    if type(sample_x_names) != list or len(sample_x_names) == 0:
        err_msg = "\"sample_x_names\" is not a valid list"
    if type(samples_y_name) != str:
        err_msg = "\"sample_y_name\" should be a string"
    
    if err_msg is None:
        try:
            for inputFileName in inputFileNames:
                with open(inputFileName, "r") as csvfile:
                    for row in csv.DictReader((row for row in csvfile if not row.startswith("#")), delimiter=",", skipinitialspace=True):
                        samples_x_values = []
                        for samples_x_name in sample_x_names:
                            samples_x_values.append(float(row[samples_x_name]))
                        
                        samples_y_value = float(row[samples_y_name])
                        
                        try:
                            idx = samples_x.index(samples_x_values)  # Checks whether we have x already
                            samples_y[idx].append(samples_y_value)
                        except ValueError:
                            samples_x.append(samples_x_values)
                            samples_y.append([samples_y_value])
            
            # Aggregate multiple observation of the sample sampling points
            samples_y_aggregation = [samples_y_aggregator(sample_y) for sample_y in samples_y]
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
