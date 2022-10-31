import os

delimitor = ','

def initialisation_pdf(path_csv) :
    print("[INIT] : Initialisation du csv : " + os.getcwd() + path_csv)
    with open(path_csv,'w') as f:
        f.write("current_time")
        f.write(delimitor)
        f.write("nb_episode")
        f.write(delimitor)
        f.write("current_episode")
        f.write(delimitor)
        f.write("discount_factor")
        f.write(delimitor)
        f.write("learning_rate")
        f.write(delimitor)
        f.write("test_frequency")
        f.write(delimitor)
        f.write("nb_tests_iteration")
        f.write(delimitor)
        f.write("epsilon_decay")
        f.write(delimitor)
        f.write("epsilon_min")
        f.write(delimitor)
        f.write("epsilon")
        f.write(delimitor)
        f.write("batch_size")
        f.write(delimitor)
        f.write("size_replay_buffer")
        f.write(delimitor)
        f.write("update_frequency")
        f.write(delimitor)
        f.write("tau")
        f.write(delimitor)
        f.write("mean_value")
        f.write(delimitor)
        f.write("std_value")
        f.write(delimitor)
        f.write("best_value")
        f.write("\n")

def write_params(path_csv, params) : 
    print("[LOG] : Writing params in csv file : " + os.getcwd() + path_csv)
    with open(path_csv,'a') as f:
        f.write(str(params["current_time"]))
        f.write(delimitor)
        f.write(str(params["nb_episode"]))
        f.write(delimitor)
        f.write(str(params["current_episode"]))
        f.write(delimitor)
        f.write(str(params["discount_factor"]))
        f.write(delimitor)
        f.write(str(params["learning_rate"]))
        f.write(delimitor)
        f.write(str(params["test_frequency"]))
        f.write(delimitor)
        f.write(str(params["nb_tests_iteration"]))
        f.write(delimitor)
        f.write(str(params["epsilon_decay"]))
        f.write(delimitor)
        f.write(str(params["epsilon_min"]))
        f.write(delimitor)
        f.write(str(params["epsilon"]))
        f.write(delimitor)
        f.write(str(params["batch_size"]))
        f.write(delimitor)
        f.write(str(params["size_replay_buffer"]))
        f.write(delimitor)
        f.write(str(params["update_frequency"]))
        f.write(delimitor)
        f.write(str(params["tau"]))
        f.write(delimitor)
        f.write(str(params["mean_value"]))
        f.write(delimitor)
        f.write(str(params["std_value"]))
        f.write(delimitor)
        f.write(str(params["best_value"]))
        f.write("\n")




def get_dict_last_params(path_csv) : 

    params = dict()
    with open(path_csv, "r") as f1:
        last_line_list = f1.readlines()[-1].split(delimitor)
        params["current_time"] = last_line_list[0]
        params["nb_episode"] = last_line_list[1]
        params["current_episode"] = last_line_list[2]
        params["discount_factor"] = last_line_list[3]
        params["learning_rate"] = last_line_list[4]
        params["test_frequency"] = last_line_list[5]
        params["nb_tests_iteration"] = last_line_list[6]
        params["epsilon_decay"] = last_line_list[7]
        params["epsilon_min"] = last_line_list[8]
        params["epsilon"] = last_line_list[9]
        params["batch_size"] = last_line_list[10]
        params["size_replay_buffer"] = last_line_list[11]
        params["update_frequency"] = last_line_list[12]
        params["tau"] = last_line_list[13]
        params["mean_value"] = last_line_list[14]
        params["std_value"] = last_line_list[15]
        params["best_value"] = last_line_list[16]
    return params

import sys

if __name__ == '__main__':


    print(sys.argv[1:])


    print(get_dict_last_params("./csv_data_parameters_flappy_bird.csv"))
