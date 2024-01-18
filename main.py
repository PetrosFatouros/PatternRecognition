import data_processing
import least_mean_squares
import least_squares
import multi_layer_neural_network


def run():
    # Print program's menu
    print('---- Main Menu ----')
    print('1 --> Least Mean Squares')
    print('2 --> Least Squares')
    print('3 --> Multi-Layer Neural Network')

    # Prompt user to select an option from the menu
    option = input('Option: ')
    while option != '1' and option != '2' and option != '3':
        option = input('Option: ')
    print()

    # Execute option
    if option == '1':
        least_mean_squares.run()
    elif option == '2':
        least_squares.run()
    elif option == '3':
        multi_layer_neural_network.run()


if __name__ == '__main__':
    # Download dataset at https://www.kaggle.com/datasets/hugomathien/soccer
    data_processing.filter_data()
    run()
