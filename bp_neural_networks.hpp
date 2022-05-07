#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

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
		for (size_t i = 0; i < _NumInputs; ++i)
		{
			_Weight.emplace_back(_RandomClamped());
		}
	}

	_Neuron(const _Neuron& neuron) noexcept
	{
		_NumInputs = neuron._NumInputs;
		_Activation = neuron._Activation;
		_Error = neuron._Error;
		_Weight = neuron._Weight;
	}

	~_Neuron() noexcept { }

protected:
	_NODISCARD double _RandomClamped() noexcept
	{
		return static_cast<double>(-1 + 2 * (rand() / (static_cast<double>(RAND_MAX) + 1)));
	}

public:
	size_t _NumInputs;	// number of neuron input
	double _Activation; // neuron output, determined by input and linear function
	double _Error;		// error
	std::vector<double> _Weight;// weight
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

	_NeuronLayer(const _NeuronLayer& layer) noexcept
	{
		_NumNeurons = layer._NumNeurons;
		_Neurons = layer._Neurons;
	}
	
	~_NeuronLayer() noexcept { }

public:
	size_t _NumNeurons; // neuron number
	std::vector<_NeuronPtr> _Neurons; // neuron pointer vector
};

template <typename _Type>
class _TrainMethodBase
{
	using _NeuronPtr = std::shared_ptr<_Neuron>;
	using _NeuronLayerPtr = std::shared_ptr<_NeuronLayer>;
public:
	_TrainMethodBase(size_t& input_layers,
		size_t& output_layers,
		size_t& hidden_layers,
		size_t& hidden_neurons,
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
		_NumHiddenLayers(hidden_layers),
		_NeuronsPerHiddenLayer(hidden_neurons),
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

	virtual bool _NetworkTraining() = 0;

	virtual bool _NetworkTrainingEpoch()
	{
		try
		{
			_ErrorSum = 0.0;
			for (size_t hid = 0; hid < _NumHiddenLayers; ++hid)
			{
				for (size_t i = 0; i < _DataIn.size(); ++i)
				{
					std::vector<double>::iterator _CurWeight;// current weight iterator
					std::vector<_NeuronPtr>::iterator _CurNeuronOut; // current neuron output iterator
					std::vector<_NeuronPtr>::iterator _CurNeuronHidden;//current hidden neuron iterator

					std::vector<_Type> _Outputs = _Update(_DataIn[i]);// update whole network

					/*Adjust the weight according to the output of each output neuron*/
					for (size_t j = 0; j < _NumOutputs; ++j)
					{
						double _Err = (_DataOut[i][j] - _Outputs[j]) * _Outputs[j] * (1 - _Outputs[j]);// error
						_ErrorSum += (_DataOut[i][j] - _Outputs[j]) * (_DataOut[i][j] - _Outputs[j]);// sum of error squares

						_NeuronLayers[hid + 1]->_Neurons[j]->_Error = _Err; // update output layer error
						_CurWeight = _NeuronLayers[hid + 1]->_Neurons[j]->_Weight.begin(); // mark first weight
						_CurNeuronHidden = _NeuronLayers[hid]->_Neurons.begin(); // mark first hidden neuron

						/*Adjust the weight according to each output neuron*/
						while (_CurWeight != _NeuronLayers[hid + 1]->_Neurons[j]->_Weight.end() - 1)
						{
							*_CurWeight += _Err * _LearningRate * (*_CurNeuronHidden)->_Activation;// adjust the weight
							_CurWeight++; // next weight
							_CurNeuronHidden++; // next hidden neuron
						}

						*_CurWeight += _Err * _LearningRate * (-1);// offset
					}

					_CurNeuronHidden = _NeuronLayers[hid]->_Neurons.begin();// the first neuron of next hidden layer

					size_t _Cnt = 0;

					/*Adjust output neuron*/
					while (_CurNeuronHidden != _NeuronLayers[hid]->_Neurons.end() - 1)
					{
						double _Err = 0;
						_CurNeuronOut = _NeuronLayers[hid + 1]->_Neurons.begin();// first neuron

						/*Adjust each weight*/
						while (_CurNeuronOut != _NeuronLayers[hid + 1]->_Neurons.end())
						{
							_Err += (*_CurNeuronOut)->_Error * (*_CurNeuronOut)->_Weight[_Cnt];
							_CurNeuronOut++;
						}

						_Err *= (*_CurNeuronHidden)->_Activation * (1 - (*_CurNeuronHidden)->_Activation);

						for (size_t j = 0; j < _NumInputs; ++j)
						{
							(*_CurNeuronHidden)->_Weight[j] += _Err * _LearningRate * _DataIn[i][j];// update hidden neuron weight
						}

						(*_CurNeuronHidden)->_Weight[_NumInputs] += _Err * _LearningRate * (-1);// offset
						_CurNeuronHidden++;
						_Cnt++;
					}
				}
			}
		}
		catch (const std::exception& e)
		{
			std::cout << "Exception: " << e.what() << std::endl;
		}

		return true;
	}

	virtual std::vector<_Type> _Update(std::vector<_Type> inputs)
	{
		std::vector<_Type> _Outputs = inputs;

		if (inputs.size() != _NumInputs)
		{
			return { };
		}

		try
		{
			for (auto& HiddenLayer : _NeuronLayers)
			{
				std::vector<_Type> _Inputs = _Outputs;
				_Outputs.clear();
				for (auto& Neuron : HiddenLayer->_Neurons)
				{
					size_t _Weight = 0;
					double _NeuronInput = 0.0;
					for (size_t k = 0; k < Neuron->_NumInputs - 1; ++k)
					{
						_NeuronInput += Neuron->_Weight[k] * _Inputs[_Weight++];
					}

					_NeuronInput += Neuron->_Weight[Neuron->_NumInputs - 1] * (-1);
					Neuron->_Activation = _Sigmoid(_NeuronInput, ACTIVATION_RESPONSE);
					_Outputs.emplace_back(static_cast<_Type>(Neuron->_Activation));
				}
			}
		}
		catch (const std::exception& e)
		{
			std::cout << "Exception: " << e.what() << std::endl;
		}

		return _Outputs;
	}

protected:

	_NODISCARD double _Sigmoid(double activation, double response) noexcept
	{
		return static_cast<double>(1 / (1 + exp(-activation / response)));
	}

protected:
	size_t& _NumInputs;// input data number
	size_t& _NumOutputs;// output data number
	size_t& _NumHiddenLayers;// hidden layer number
	size_t& _NeuronsPerHiddenLayer;//hidden layer neuron
	size_t& _NumEpochs;// epochs
	double& _LearningRate;// learn rate
	double& _ErrorSum; // error summary
	double& _ErrorThresHold; // error threshold
	long& _TrainEpochs;     // train times
	bool& _Trained;
	bool _Debug;
	std::vector<_NeuronLayerPtr>& _NeuronLayers;
	std::vector<std::vector<_Type>>& _DataIn;
	std::vector<std::vector<_Type>>& _DataOut;
};

template <typename _Type>
class _TrainByEpochs : public _TrainMethodBase<_Type>
{
	using _NeuronLayerPtr = std::shared_ptr<_NeuronLayer>;
public:
	_TrainByEpochs(size_t& input_layers,
		size_t& output_layers,
		size_t& hidden_layers,
		size_t& hidden_neurons,
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
		_TrainMethodBase<_Type>(input_layers, 
			output_layers, 
			hidden_layers,
			hidden_neurons,
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

	virtual bool _NetworkTraining()
	{
		long cnt = _TrainMethodBase<_Type>::_TrainEpochs;
		while (cnt--)
		{
			if (_TrainMethodBase<_Type>::_Debug)
			{
				std::cout << "ErrorSum: " << std::setprecision(12) << _TrainMethodBase<_Type>::_ErrorSum << std::endl;
			}

			_TrainMethodBase<_Type>::_NetworkTrainingEpoch();
		}

		_TrainMethodBase<_Type>::_Trained = true;
		return true;
	}
};

template <typename _Type>
class _TrainByErrorSum : public _TrainMethodBase<_Type>
{
	using _NeuronLayerPtr = std::shared_ptr<_NeuronLayer>;
public:
	_TrainByErrorSum(size_t& input_layers,
		size_t& output_layers,
		size_t& hidden_layers,
		size_t& hidden_neurons,
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
		_TrainMethodBase<_Type>(input_layers,
			output_layers,
			hidden_layers,
			hidden_neurons,
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

	virtual bool _NetworkTraining()
	{
		while (_TrainMethodBase<_Type>::_ErrorSum > _TrainMethodBase<_Type>::_ErrorThresHold)
		{
			if (_TrainMethodBase<_Type>::_Debug)
			{
				std::cout << "ErrorSum: " << std::setprecision(12) << _TrainMethodBase<_Type>::_ErrorSum << std::endl;
			}

			_TrainMethodBase<_Type>::_NetworkTrainingEpoch();
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
		size_t hidden_layers,
		size_t hidden_neurons,
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
		_NumHiddenLayers(hidden_layers),
		_NeuronsPerHiddenLayer(hidden_neurons),
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

	void push_data(std::vector<std::vector<_Type>>& indata, std::vector<std::vector<_Type>>& outdata) noexcept
	{
		_DataIn = indata;
		_DataOut = outdata;
	}

	bool train()
	{
		return _TrainMethodPtr->_NetworkTraining();
	}
	
	std::vector<_Type> recognition(std::vector<_Type>& indata)
	{
		return _TrainMethodPtr->_Update(indata);
	}

	void save(std::string filename) noexcept
	{
		std::ofstream _WriteData(filename);
		std::string _Output;
		std::stringstream _StrStream;

		_Output = std::to_string(this->_NumInputs);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_NumOutputs);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_NumHiddenLayers);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_NeuronsPerHiddenLayer);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_NumEpochs);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_LearningRate);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_ErrorSum);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_ErrorThresHold);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_TrainEpochs);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_Debug);
		_WriteData << _Output << std::endl;
		_Output = std::to_string(this->_Trained);
		_WriteData << _Output << std::endl;

		for (auto& HidLayer : this->_NeuronLayers)
		{
			for (auto& Neuron : HidLayer->_Neurons)
			{
				for (auto& Weight : Neuron->_Weight)
				{
					_StrStream.str(std::string());
					_StrStream << std::setprecision(18) << Weight;
					_WriteData << _StrStream.str() << std::endl;
				}
			}
		}
		
		_WriteData.clear();
		_WriteData.close();
	}

	void load(std::string filename) noexcept
	{
		char _Buffer[128] = { 0 };
		std::ifstream _ReadData(filename);
		if (_ReadData.is_open())
		{
			_ReadData.getline(_Buffer, 128);
			this->_NumInputs = _StringToNum<size_t>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_NumOutputs = _StringToNum<size_t>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_NumHiddenLayers = _StringToNum<size_t>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_NeuronsPerHiddenLayer = _StringToNum<size_t>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_NumEpochs = _StringToNum<size_t>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_LearningRate = _StringToNum<double>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_ErrorSum = _StringToNum<double>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_ErrorThresHold = _StringToNum<double>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_TrainEpochs = _StringToNum<long>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_Debug = _StringToNum<bool>(_Buffer);
			_ReadData.getline(_Buffer, 128);
			this->_Trained = _StringToNum<bool>(_Buffer);
			
			for (auto& HidLayer : this->_NeuronLayers)
			{
				for (auto& Neuron : HidLayer->_Neurons)
				{
					for (auto& Weight : Neuron->_Weight)
					{
						_ReadData.getline(_Buffer, 128);
						Weight = _StringToNum<double>(_Buffer);
					}
				}
			}
		}

		_ReadData.clear();
		_ReadData.close();
	}

private:
	void _CreateNetworks() noexcept
	{
		if (_NumHiddenLayers > 0) 
		{
			_NeuronLayers.emplace_back(std::make_shared<_NeuronLayer>(_NeuronsPerHiddenLayer, _NumInputs));		//first hidden layer

			for (size_t i = 0; i < _NumHiddenLayers - 1; ++i) 
			{
				_NeuronLayers.emplace_back(std::make_shared<_NeuronLayer>(_NeuronsPerHiddenLayer, _NeuronsPerHiddenLayer));//other hidden layer
			}

			_NeuronLayers.emplace_back(std::make_shared<_NeuronLayer>(_NumOutputs, _NeuronsPerHiddenLayer));		// output
		}
		else
		{
			_NeuronLayers.emplace_back(std::make_shared<_NeuronLayer>(_NumOutputs, _NumInputs));
		}
	}

	_NODISCARD double _RandomClamped() noexcept
	{
		return static_cast<double>(-1 + 2 * (rand() / (static_cast<double>(RAND_MAX) + 1)));
	}

	template <typename _T>
	_NODISCARD _T _StringToNum(const std::string& str) noexcept
	{
		std::istringstream iss(str);
		_T num;
		iss >> num;
		return num;
	}

private:
	size_t _NumInputs;// input data
	size_t _NumOutputs;// output data
	size_t _NumHiddenLayers;// hidden layer number
	size_t _NeuronsPerHiddenLayer;// hidden layer neuron
	size_t _NumEpochs;// epochs
	double _LearningRate;// learning rate
	double _ErrorSum;// error summary
	double _ErrorThresHold; // error threshold
	long _TrainEpochs; // train times 
	bool _Debug;
	bool _Trained;

	std::vector<_NeuronLayerPtr> _NeuronLayers;//²ãÊý
	std::vector<std::vector<_Type>> _DataIn;
	std::vector<std::vector<_Type>> _DataOut;
	std::shared_ptr<_TrainMethodBase<_Type>> _TrainMethodPtr;
};