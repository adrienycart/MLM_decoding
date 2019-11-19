import weight_search

import argparse
import os
import sys
import pickle



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, choices=["time", "quant", "event"], help="Change the step type " +
                        "for frame timing. Either time (default), quant (for 16th notes), or event (for onsets).",
                        default="time")
    parser.add_argument("-o", "--output", help="The directory to save outputs to. Defaults to None (don't save).",
                        default=None)
    parser.add_argument("--num", help="The number of tests to do for each data point. Defaults to 1.", type=int,
                        default=1)
    parser.add_argument("--gpu", help="The gpu to use. Defaults to 0.", default="0")
    args = parser.parse_args()
    
    print("Running " + str(args.num) + " times per data point.")
    print("step type: " + args.step)
    print("saving output to " + args.output)
    print("using GPU " + args.gpu)
    sys.stdout.flush()
    
    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    weight_search.set_step(args.step)
    weight_search.load_data()
    weight_search.load_model()
    
    best_data = None
    best_num = None
    best_f_n = 0.0
    best_model = None
    
    for gt in [False]:
        for min_diff in [0]:
            for history in range(5, 41, 5) if args.step == "time" else range(5,11):
                for num_layers in range(1, 4):
                    for is_weight in [False]:
                        for features in [False]:
                            for prior_context in [0]:
                                for history_context in [0]:
                                    for use_lstm in [False]:
                                        for num in range(args.num):
                                            data = [gt, min_diff, history, num_layers, is_weight, features,
                                                    history_context, prior_context, use_lstm]
                                            f_n = -weight_search.weight_search(data, num=num)

                                            if f_n > best_f_n:
                                                best_data = data
                                                best_f_n = f_n
                                                best_num = num
                                                best_model = weight_search.get_most_recent_model()

                                                with open(args.output + "/best_model.pkl", "wb") as file:
                                                    pickle.dump(best_model, file)

    print("Best data: " + str(best_data))
    print("Best f_n: " + str(best_f_n))
    print("Best num: " + str(best_num))
