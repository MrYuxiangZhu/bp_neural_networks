#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#define ACTIVATION_RESPONSE 1.0

#ifndef _NODISCARD
#define _NODISCARD [[nodiscard]]
#endif

enum class STOP_TYPE 
{ 
	COUNT = 0, 
	ERROR_SUM,
};

class _Neuron
{
public:
	_Neuron(size_t inputs) noexcept : _NumInputs(inputs + 1), _Activation(0.0), _Error(0.0)
	{
		for (size_t i = 0; i < _NumInputs + 1; ++i)
		{
			_Weight.emplace_back(_RandomClamped());
		}
	}

	~_Neuron() noexcept { }

protected:
	_NODISCARD double _RandomClamped() noexcept
	{
		return static_cast<double>(-1 + 2 * (rand() / ((double)RAND_MAX + 1)));
	}

public:
	size_t _NumInputs;
	double _Activation;
	double _Error;
	std::vector<double> _Weight;
};

class _NeuronLayer
{
	using _NeuronPtr = std::shared_ptr<_Neuron>;
public:
	_NeuronLayer(size_t neurons, size_t inputs) noexcept : _NumNeurons(neurons)
	{
		for (size_t i = 0; i < _NumNeurons; ++i)
		{
			_Neurons.emplace_back(std::make_shared<_Neuron>(inputs));
		}
	}
	
	~_NeuronLayer() noexcept { }

public:
	size_t _NumNeurons;
	std::vector<_NeuronPtr> _Neurons;
};

template <typename _Type>
class _TrainMethodBase
{
	using _NeuronPtr = std::shared_ptr<_Neuron>;
	using _NeuronLayerPtr = std::shared_ptr<_NeuronLayer>;
public:
	_TrainMethodBase(size_t& input_layers,
		size_t& output_layers,
		size_t& hidden_neurons,
		size_t& hidden_layers_num,
		double& learning_rate,
		double& error_threshold,
		long& train_epochs,
		double& error_sum,
		bool& trained,
		size_t& num_epochs, 
		bool& debug,
		std::vector<_NeuronLayerPtr>& layers, 
		std::vector<std::vector<_Type>>& input_data, 
		std::vector<std::vector<_Type>>& output_data) noexcept :
		_NumInputs(input_layers),
		_NumOutputs(output_layers),
		_NumHiddenLayers(hidden_neurons),
		_NeuronsPerHiddenLayer(hidden_layers_num),
		_LearningRate(learning_rate),
		_ErrorThresHold(error_threshold),
		_TrainEpochs(train_epochs),
		_ErrorSum(error_sum),
		_Trained(trained),
		_NumEpochs(num_epochs),
		_Debug(debug),
		_NeuronLayers(layers),
		_DataIn(input_data),
		_DataOut(output_data)
	{
		
	}

	virtual ~_TrainMethodBase() noexcept { }

	virtual bool _NetworkTraining() noexcept = 0;

protected:
	bool _NetworkTrainingEpoch() noexcept
	{
		_ErrorSum = 0.0;//置零
		std::vector<double>::iterator cur_weight;//指向某个权重
		std::vector<_NeuronPtr>::iterator cur_neuron_out; //指向输出神经元
		std::vector<_NeuronPtr>::iterator cur_neuron_hidden;//某个隐藏神经元
		//对每一个输入集合调整权值
		for (size_t vec = 0; vec < _DataIn.size(); ++vec)
		{
			std::vector<_Type> outputs = _Update(_DataIn[vec]);//通过神经网络获得输出
			//根据每一个输出神经元的输出调整权值
			for (size_t op = 0; op < _NumOutputs; ++op)
			{
				double err = (_DataOut[vec][op] - outputs[op]) * outputs[op] * (1 - outputs[op]);//误差的平方
				_ErrorSum += (_DataOut[vec][op] - outputs[op]) * (_DataOut[vec][op] - outputs[op]);//计算误差总和，用于暂停训练
				_NeuronLayers[1]->_Neurons[op]->_Error = err;//更新误差（输出层）
				cur_weight = _NeuronLayers[1]->_Neurons[op]->_Weight.begin();//标记第一个权重
				cur_neuron_hidden = _NeuronLayers[0]->_Neurons.begin();//标记隐藏层第一个神经元
				//对该神经元的每一个权重进行调整
				while (cur_weight != _NeuronLayers[1]->_Neurons[op]->_Weight.end() - 1)
				{
					*cur_weight += err * _LearningRate * (*cur_neuron_hidden)->_Activation;//根据误差和学习率和阈值调整权重
					cur_weight++;//指向下一个权重
					cur_neuron_hidden++;//指向下一个隐藏层神经元
				}
				*cur_weight += err * _LearningRate * (-1);//偏移值
			}

			cur_neuron_hidden = _NeuronLayers[0]->_Neurons.begin();//重新指向隐藏层第一个神经元
			int n = 0;
			//对每一个隐藏层神经元
			while (cur_neuron_hidden != _NeuronLayers[0]->_Neurons.end() - 1)
			{
				float err = 0;
				cur_neuron_out = _NeuronLayers[1]->_Neurons.begin();//指向第一个输出神经元
				//对每一个输出神经元的第一个权重
				while (cur_neuron_out != _NeuronLayers[1]->_Neurons.end())
				{
					err += (*cur_neuron_out)->_Error * (*cur_neuron_out)->_Weight[n];//某种计算误差的方法(BP)
					cur_neuron_out++;
				}
				err *= (*cur_neuron_hidden)->_Activation * (1 - (*cur_neuron_hidden)->_Activation);//某种计算误差的方法(BP)
				for (size_t w = 0; w < _NumInputs; ++w)
				{
					(*cur_neuron_hidden)->_Weight[w] += err * _LearningRate * _DataIn[vec][w];//根据误差更新隐藏层的权重
				}
				(*cur_neuron_hidden)->_Weight[_NumInputs] += err * _LearningRate * (-1);//偏移值
				cur_neuron_hidden++;//下一个隐藏层神经元
				n++;//下一个权重
			}
		}

		return true;
	}

	template <typename _T>
	_NODISCARD std::vector<_T> _Update(std::vector<_T> inputs) noexcept
	{
		std::vector<_T> outputs;
		size_t weight = 0;

		if (inputs.size() != _NumInputs)
		{
			return outputs;
		}

		for (size_t i = 0; i < _NumHiddenLayers + 1; ++i) 
		{
			if (i > 0)
			{
				inputs = outputs;
			}

			outputs.clear();
			weight = 0;
			for (size_t n = 0; n < _NeuronLayers[i]->_NumNeurons; ++n)
			{
				double netinput = 0.0;
				size_t numInputs = _NeuronLayers[i]->_Neurons[n]->_NumInputs;

				for (int k = 0; k < numInputs - 1; ++k) 
				{
					netinput += _NeuronLayers[i]->_Neurons[n]->_Weight[k] * inputs[weight++];
				}

				netinput += _NeuronLayers[i]->_Neurons[n]->_Weight[numInputs - 1] * (-1);
				_NeuronLayers[i]->_Neurons[n]->_Activation = _Sigmoid(netinput, ACTIVATION_RESPONSE);
				outputs.emplace_back(_NeuronLayers[i]->_Neurons[n]->_Activation);
				weight = 0;
			}
		}

		return outputs;
	}

	_NODISCARD double _Sigmoid(double activation, double response) noexcept
	{
		return static_cast<double>(1 / (1 + exp(-activation / response)));
	}

protected:
	size_t& _NumInputs;//输入量
	size_t& _NumOutputs;//输出量
	size_t& _NumHiddenLayers;//隐藏层数
	size_t& _NeuronsPerHiddenLayer;//隐藏层拥有的神经元
	size_t& _NumEpochs;//代数
	double& _LearningRate;//学习率
	double& _ErrorSum;//误差总值
	double& _ErrorThresHold;     //误差阈值（什么时候停止训练）
	long& _TrainEpochs;     //训练次数（什么时候停止训练）
	bool& _Debug;//是否输出误差值
	bool& _Trained;//是否已经训练过
	std::vector<_NeuronLayerPtr>& _NeuronLayers;
	std::vector<std::vector<_Type>>& _DataIn;
	std::vector<std::vector<_Type>>& _DataOut;
};

template <typename _Type>
class _TrainByEpochs : public _TrainMethodBase<_Type>
{
	using _NeuronLayer = typename _TrainMethodBase<_Type>::_NeuronLayerPtr;
public:
	_TrainByEpochs(size_t& input_layers,
		size_t& output_layers,
		size_t& hidden_neurons,
		size_t& hidden_layers_num,
		double& learning_rate,
		double& error_threshold,
		long& train_epochs,
		double& error_sum,
		bool& trained,
		size_t& num_epochs,
		bool& debug,
		std::vector<_NeuronLayer>& layers, 
		std::vector<std::vector<_Type>>& input_data, 
		std::vector<std::vector<_Type>>& output_data) noexcept :
		_TrainMethodBase<_Type>(input_layers, 
			output_layers, 
			hidden_neurons, 
			hidden_layers_num, 
			learning_rate, 
			error_threshold,
			train_epochs,
			error_sum,
			trained,
			num_epochs,
			debug,
			layers, 
			input_data, 
			output_data)
	{

	}

	virtual ~_TrainByEpochs() noexcept { }

	virtual bool _NetworkTraining() noexcept
	{
		long cnt = _TrainMethodBase<_Type>::_TrainEpochs;
		while (cnt--)
		{
			if (_TrainMethodBase<_Type>::_Debug)
			{
				std::cout << "ErrorSum:" << _TrainMethodBase<_Type>::_ErrorSum << std::endl;
			}

			_NetworkTraining();
		}

		_TrainMethodBase<_Type>::_Trained = true;
		return true;
	}
};

template <typename _Type>
class _TrainByErrorSum : public _TrainMethodBase<_Type>
{
	using _NeuronLayer = typename _TrainMethodBase<_Type>::_NeuronLayerPtr;
public:
	_TrainByErrorSum(size_t& input_layers,
		size_t& output_layers,
		size_t& hidden_neurons,
		size_t& hidden_layers_num,
		double& learning_rate,
		double& error_threshold,
		long& train_epochs,
		double& error_sum,
		bool& trained,
		size_t& num_epochs,
		bool& debug,
		std::vector<_NeuronLayer>& layers,
		std::vector<std::vector<_Type>>& input_data,
		std::vector<std::vector<_Type>>& output_data) noexcept :
		_TrainMethodBase<_Type>(input_layers,
			output_layers,
			hidden_neurons,
			hidden_layers_num,
			learning_rate,
			error_threshold,
			train_epochs,
			error_sum,
			trained,
			num_epochs,
			debug,
			layers,
			input_data,
			output_data)
	{

	}

	virtual ~_TrainByErrorSum() noexcept { }

	virtual bool _NetworkTraining() noexcept
	{
		while (_TrainMethodBase<_Type>::_ErrorSum > _TrainMethodBase<_Type>::_ErrorThresHold)
		{
			if (_TrainMethodBase<_Type>::_Debug)
			{
				std::cout << "ErrorSum:" << _TrainMethodBase<_Type>::_ErrorSum << std::endl;
			}

			_NetworkTraining();
		}

		_TrainMethodBase<_Type>::_Trained = true;
		return true;
	}
};

template <typename _Type>
class bp_neural_networks
{
	using _NeuronPtr = std::shared_ptr<_Neuron>;
	using _NeuronLayerPtr = std::shared_ptr<_NeuronLayer>;
public:
	bp_neural_networks(size_t input_layers,
		size_t output_layers,
		size_t hidden_neurons,
		size_t hidden_layers_num,
		double learning_rate,
		double error_threshold,
		long train_epochs,
		double error_sum,
		bool trained,
		size_t num_epochs,
		STOP_TYPE stop_type = STOP_TYPE::COUNT,
		bool debug = false) noexcept :
		_NumInputs(input_layers),
		_NumOutputs(output_layers),
		_NumHiddenLayers(hidden_neurons),
		_NeuronsPerHiddenLayer(hidden_layers_num),
		_LearningRate(learning_rate),
		_ErrorThresHold(error_threshold),
		_TrainEpochs(train_epochs),
		_ErrorSum(error_sum),
		_Trained(trained),
		_NumEpochs(num_epochs)
	{
		_CreateNetworks();

		switch (stop_type)
		{
		case STOP_TYPE::COUNT:
			_TrainMethodPtr = std::make_shared<_TrainByEpochs<_Type>>(_NumInputs, _NumOutputs, _NumHiddenLayers, \
																	_NeuronsPerHiddenLayer, _LearningRate, _ErrorThresHold, \
																	_TrainEpochs, _ErrorSum, _Trained, _NumEpochs, debug, \
																	_NeuronLayers, _DataIn, _DataOut);
			break;
		case STOP_TYPE::ERROR_SUM:
			_TrainMethodPtr = std::make_shared<_TrainByErrorSum<_Type>>(_NumInputs, _NumOutputs, _NumHiddenLayers, \
																	_NeuronsPerHiddenLayer, _LearningRate, _ErrorThresHold, \
																	_TrainEpochs, _ErrorSum, _Trained, _NumEpochs, debug, \
																	_NeuronLayers, _DataIn, _DataOut);
			break;
		default:
			break;
		}
	}

	~bp_neural_networks() noexcept { }

	void push_data(std::vector<_Type>& indata, std::vector<_Type>& outdata) noexcept
	{
		_DataIn.emplace_back(indata);
		_DataOut.emplace_back(outdata);
	}

	bool train()
	{
		return _TrainMethodPtr->_NetworkTraining();
	}

	void save()
	{
		//ofstream WriteData(filename);
		//string output;
		//output = to_string(this->EachNums);
		//WriteData << output << endl;
		//output = to_string(this->FrameNums);
		//WriteData << output << endl;
		//output = to_string(this->VideoNumVectorSize);
		//WriteData << output << endl;
		//output = to_string(this->VideoTypeVectorSize);
		//WriteData << output << endl;
		//for (int i = 0; i < VideoTypeVectorSize; ++i) {
		//	for (int j = 0; j < VideoNumVectorSize * FrameNums; ++j) {
		//		for (int k = 0; k < EachNums; ++k) {
		//			output = to_string(this->SetIn[i][j][k]);
		//			WriteData << output << endl;
		//		}
		//	}
		//}
		//for (int i = 0; i < VideoTypeVectorSize; ++i) {
		//	for (int j = 0; j < VideoNumVectorSize * FrameNums; ++j) {
		//		for (int k = 0; k < VideoTypeVectorSize; ++k) {
		//			output = to_string(this->SetOut[i][j][k]);
		//			WriteData << output << endl;
		//		}
		//	}
		//}
		//WriteData.clear(); //为了代码具有移植性和复用性, 这句最好带上,清除标志位.有些系统若不清理可能会出现问题.
		//WriteData.close();
	}

	void load()
	{
		//char buffer[100];
		//ifstream ReadData(filename);
		//if (ReadData.is_open())
		//{
		//	ReadData.getline(buffer, 100);
		//	this->EachNums = stringToNum<int>(buffer);
		//	ReadData.getline(buffer, 100);
		//	this->FrameNums = stringToNum<int>(buffer);
		//	ReadData.getline(buffer, 100);
		//	this->VideoNumVectorSize = stringToNum<int>(buffer);
		//	ReadData.getline(buffer, 100);
		//	this->VideoTypeVectorSize = stringToNum<int>(buffer);
		//	for (int i = 0; i < VideoTypeVectorSize; ++i) {
		//		for (int j = 0; j < VideoNumVectorSize * FrameNums; ++j) {
		//			for (int k = 0; k < EachNums; ++k) {
		//				ReadData.getline(buffer, 100);
		//				this->SetIn[i][j][k] = stringToNum<float>(buffer);
		//			}
		//		}
		//	}
		//	for (int i = 0; i < VideoTypeVectorSize; ++i) {
		//		for (int j = 0; j < VideoNumVectorSize * FrameNums; ++j) {
		//			for (int k = 0; k < VideoTypeVectorSize; ++k) {
		//				ReadData.getline(buffer, 100);
		//				this->SetOut[i][j][k] = stringToNum<float>(buffer);
		//			}
		//		}
		//	}
		//}
		//ReadData.clear(); //为了代码具有移植性和复用性, 这句最好带上,清除标志位.有些系统若不清理可能会出现问题.
		//ReadData.close();
	}

private:
	void _CreateNetworks() 
	{
		if (_NumHiddenLayers > 0) 
		{
			_NeuronLayers.emplace_back(_NeuronLayer(_NeuronsPerHiddenLayer, _NumInputs));		//输入层
			for (int i = 0; i < _NumHiddenLayers - 1; ++i) 
			{
				_NeuronLayers.emplace_back(_NeuronLayer(_NeuronsPerHiddenLayer, _NeuronsPerHiddenLayer));//隐层
			}
			_NeuronLayers.emplace_back(_NeuronLayer(_NumOutputs, _NeuronsPerHiddenLayer));		//输出层
		}
		else 
		{
			_NeuronLayers.emplace_back(_NeuronLayer(_NumOutputs, _NumInputs));
		}

		for (size_t i = 0; i < _NumHiddenLayers + 1; ++i) //权值维数比隐层数多1
		{								
			for (size_t j = 0; j < _NeuronLayers[i]->_NumNeurons; ++j)
			{
				for (size_t k = 0; k < _NeuronLayers[i]->_Neurons[j]->_NumInputs; ++k)
				{
					_NeuronLayers[i]->_Neurons[j]->_Weight[k] = _RandomClamped();//随机生成权重
				}
			}
		}
	}

	_NODISCARD double _RandomClamped()
	{
		return static_cast<double>(-1 + 2 * (rand() / ((double)RAND_MAX + 1)));
	}

	template <typename _T>
	_NODISCARD _T _StringToNum(const std::string& str)
	{
		std::istringstream iss(str);
		_T num;
		iss >> num;
		return num;
	}

private:
	size_t _NumInputs;//输入量
	size_t _NumOutputs;//输出量
	size_t _NumHiddenLayers;//隐藏层数
	size_t _NeuronsPerHiddenLayer;//隐藏层拥有的神经元
	size_t _NumEpochs;//代数
	double _LearningRate;//学习率
	double _ErrorSum;//误差总值
	double _ErrorThresHold;     //误差阈值（什么时候停止训练）
	long _TrainEpochs;     //训练次数（什么时候停止训练）
	bool _Debug;//是否输出误差值
	bool _Trained;//是否已经训练过

	std::vector<_NeuronLayerPtr> _NeuronLayers;//层数
	std::vector<std::vector<_Type>> _DataIn;
	std::vector<std::vector<_Type>> _DataOut;
	std::shared_ptr<_TrainMethodBase<_Type>> _TrainMethodPtr;
};