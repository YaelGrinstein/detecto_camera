import os
import matplotlib.pyplot as plt
import argparse
from detecto_core import Dataset, Model
from evaluation import evaluate, predict_from_foscam

LABELS = ["7290002057123_olives", "7290113192393_olive-mix", "7290000307046_garden_peas_and_carrots",
                 "7290000307237_sweet_corn", "7290113192546_cucumbers_in_vinegar", "8005110170300_cut_tomatos",
                 "7290000208428_humus_yachein", "7290012453182_cucumbers_in_brin", "7290113192355_green_olives_beit",
                 "7290113192539_sult_pickles_beit", "8005110551253_pizza_sauce_mutti"]


#LABELS = [ "7290113192393_olive-mix", "7290000307046_garden_peas_and_carrots",
 #                "7290000307237_sweet_corn", "7290113192546_cucumbers_in_vinegar", "8005110170300_cut_tomatos",
  #               "7290000208428_humus_yachein", "7290113192355_green_olives_beit",
   #              "7290113192539_sult_pickles_beit", "8005110551253_pizza_sauce_mutti"]
# Arguments
def parse_args():
    """Parse script arguments.

    Get training hyper-parameters such as: learning rate, momentum,
    batch size, number of training epochs and optimizer.
    Get training dataset and the model name.
    Get evaluation method
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval', type=int, default=0,  help='Evaluate models, default <0>')
    parser.add_argument('-t', '--train', type=int, default=1, help='train detecto model, default <1>')
    parser.add_argument('--epochs', default=10, type=int,help='Number of epochs to run')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate, default <1e-2>')
    parser.add_argument('-SGD', '--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('-w', '--weight_decay', default=0.0005, type=int, help='weight decay')
    parser.add_argument('-lrs', '--learning_step_size', default=3, type=int, help='learning_step_size')
    parser.add_argument('-g', '--gamma', default=3, type=int, help='gamma')
    parser.add_argument('-b', '--batch_size', default=2, type=int,  help='Training batch size')
    parser.add_argument('-fsc', '--foscam', type=bool, default=False, help='data source from live camera foscam')
    parser.add_argument('-server','--run_server', type=bool, default=False, help='run with server')
    parser.add_argument('-m', '--model',  default='detecto_model_idans_data.pth', type=str, help='Model name')
    parser.add_argument('-oos', '--outs_of_sample',  default=False, type=str, help='evaluate over out of samlple data ')


    return parser.parse_args()


def main():

    args = parse_args()

    #Data
    main_dir = os.getcwd()
    train_dir = os.path.join(main_dir, "train_dir")
    val_dir = os.path.join(main_dir, "val_dir")
    test_dir = os.path.join(main_dir, "test_dir")
    prediction_dir = os.path.join(main_dir, "predictions")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    train_data = Dataset(train_dir)
    val_data = Dataset(val_dir)

    #Train
    if args.train:
        model = Model(LABELS)
        losses = model.fit(train_data, val_data, args.epochs, args.learning_rate, args.momentum, args.weight_decay,
                          args.gamma, args.learning_step_size)
        plt.plot(losses)
        plt.show()
        model.save(args.model)

    #Evaluation
    model_path = os.path.join(main_dir, args.model)
    if args.eval:
        evaluate(model_path, test_dir, LABELS, prediction_dir, OutOfSample=False)
    if args.out_of_sample:
        out_of_sample = os.path.join(main_dir, "out_of_sample")
        evaluate(model_path, out_of_sample, LABELS, prediction_dir, OutOfSample=True)

    if args.foscam:
        predict_from_foscam(args.model, LABELS, prediction_dir)


if __name__ == '__main__':
    main()