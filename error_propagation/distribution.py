import numpy as np
import matplotlib.pyplot as plt

# input_sample_size = 10
# output_sample_size = 10
test_size = 5000
scale = 1

def model(input):
    return input

def model_error(input):
    output_noise = np.random.normal(1,1*scale,len(input))
    # output_noise = np.random.uniform(-1,1,len(input))
    return model(input)+output_noise

def main(input_sample_size, output_sample_size):
    input_error_distribution = np.random.normal(1,1*scale,input_sample_size)
    # input_error_distribution = np.random.uniform(-1,1,input_sample_size)

    # first, get output error distribution
    test_input = np.arange(0,output_sample_size*scale*0.1,0.1*scale)
    test_output = model_error(test_input)
    ground_truth = model(test_input)
    output_error_distribution = test_output - ground_truth

    inside = 0
    for i in range(test_size):
        output_total = []
        input = np.random.randint(-99999,99999)
        # input = input / 1000
        inputs = input+input_error_distribution
        outputs = model_error(inputs)
        for x in outputs:
            output_total.append(output_error_distribution+x)
        output_total = np.array(output_total)
        output_total.reshape((-1))
        # print(output_total)
        avg = np.mean(output_total)
        std = np.std(output_total)
        lower_bound = avg-1.96*std
        upper_bound = avg+1.96*std
        truth = model(input)
        # print(lower_bound,upper_bound,truth)
        if lower_bound <= truth and truth <= upper_bound:
            inside += 1
        # plt.hist(output_total)
        # plt.title("ground_truth="+str(truth))
        # plt.savefig('histogram.jpg')

    # print(inside,test_size)
    return inside/test_size


# main()
for k in range(1):
    res = []
    input_sample_size = 100
    output_sample_size = 100
    print(input_sample_size,output_sample_size,test_size,scale)
    for i in range(100):
        res.append(main(input_sample_size, output_sample_size))
    print(np.mean(res))
    print(np.std(res))