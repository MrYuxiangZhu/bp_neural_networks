#include <iostream>
#include "bp_neural_networks.hpp"

int main()
{
    std::vector<std::vector<double>> input_data;
    std::vector<std::vector<double>> output_data;

    // train data
    std::vector<double> input1;
    input1.push_back(2.0);
    input1.push_back(1.0);

    std::vector<double> output1;
    output1.push_back(0.0);

    input_data.push_back(input1);
    output_data.push_back(output1);

    std::vector<double> input2;
    input2.push_back(1.0);
    input2.push_back(-1.0);

    std::vector<double> output2;
    output2.push_back(1.0);

    input_data.push_back(input2);
    output_data.push_back(output2);

    std::vector<double> input3;
    input3.push_back(-1.0);
    input3.push_back(-1.0);

    std::vector<double> output3;
    output3.push_back(0.0);

    input_data.push_back(input3);
    output_data.push_back(output3);

    std::vector<double> input4;
    input4.push_back(-1.0);
    input4.push_back(1.0);

    std::vector<double> output4;
    output4.push_back(1.0);

    input_data.push_back(input4);
    output_data.push_back(output4);

    bp_neural_networks<double> bp(2, 1, 2, 20, 0.1, 0.01, 10000, 1, false, 0, STOP_TYPE::COUNT, true);
    bp.push_data(input_data, output_data);
    bool trained = bp.train();
    std::vector<double> res = bp.recognition(input2);
    //bp.save("test.txt");
    //bp.load("test.txt");
    return 0;
}