#ifndef CMGRDF_OnnxDNNEvaluator_h
#define CMGRDF_OnnxDNNEvaluator_h

#include <vector>
#include <array>
#include <string>
#include <exception>
#include <cassert>
#include <iostream>
#include <ROOT/RVec.hxx>

#include "onnxruntime_cxx_api.h"

class OnnxDNNEvaluator {
public:
  OnnxDNNEvaluator(const char *modelFile, bool verbose = false, const std::string& taggerType = "default")
      : modelFile_(modelFile),
        taggerType_(taggerType),
        env_(ORT_LOGGING_LEVEL_WARNING, modelFile),
        options_(defaultOpts()),
        memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
        session_(env_, modelFile, options_) {
    
    if (verbose)
      std::cout << "Loading Onnx file " << modelFile << std::endl;
      
    if (session_.GetInputCount() != 1)
      throw std::logic_error("This works only with a single input tensor");
    if (session_.GetOutputCount() != 1)
      throw std::logic_error("This works only with a single output tensor");
      
    inputName_ = std::string(session_.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get());
    //std::cout << "inputName_  ==> \t" << inputName_ << std::endl; 
    outputName_ = std::string(session_.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get());
    //std::cout << "outputName_ ==> \t" << outputName_ << std::endl;
    inputNameC_[0] = inputName_.c_str();
    //std::cout << "inputNameC_[0]  ==> \t" << inputNameC_[0] << std::endl;
    outputNameC_[0] = outputName_.c_str();
    //std::cout << "outputNameC_[0] ==> \t" << outputNameC_[0] << std::endl;
    
    auto inputTypeInfo = session_.GetInputTypeInfo(0);
    //std::cout << "inputTypeInfo :" << inputTypeInfo << std::endl;
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();  // don't merge with above
    //std::cout << "inputTensorInfo :" << inputTensorInfo << std::endl;
    auto inputShape = inputTensorInfo.GetShape();
    //std::cout << "Input Shape Size :" << inputShape.size() << std::endl;
    //std::cout << "Zeroth Index Input Shape :" << inputShape[0] << std::endl;
    
    auto outputTypeInfo = session_.GetOutputTypeInfo(0);
    //std::cout << "outputTypeInfo :" << outputTypeInfo << std::endl;
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();  // don't merge with above
    //std::cout << "outputTensorInfo :" << outputTensorInfo << std::endl;
    auto outputShape = outputTensorInfo.GetShape();
    //std::cout << "Output Shape Size :" << outputShape.size() << std::endl;
    //std::cout << "Zeroth Index Output Shape :" << outputShape[0] << std::endl;
    
    // Handle shape checking based on tagger type
    if (taggerType_ == "hjtagger") {
        if (inputShape.size() != 2 || inputShape[0] != -1)
            throw std::logic_error("Hjtagger : Input inputShape size is expected to be (-1, N)");
        if (outputShape.size() != 2 || outputShape[0] != -1)
            throw std::logic_error("Hjtagger : Output outputShape size is expected to be (-1, N)");
    } else if (taggerType_ == "hadtoptagger") {
        if (inputShape.size() != 2 || (inputShape[0] != -1 && inputShape[0] != 1))
            throw std::logic_error("Hadtoptagger : Input inputShape size is expected to be (-1, N) or (1, N)");
        if (outputShape.size() != 2 || (outputShape[0] != -1 && outputShape[0] != 1))
            throw std::logic_error("Hadtoptagger : Output outputShape size is expected to be (-1, N) or (1, N)");
    } else {
        throw std::logic_error("Unknown tagger type: " + taggerType_);
    }
    
    nInputs_ = inputShape[1];
    //std::cout << "First Index Input Shape :" << nInputs_ << std::endl;
    nOutputs_ = outputShape[1];
    //std::cout << "First Index Output Shape :" << nOutputs_ << std::endl;
    inputShape_[0] = 1;
    inputShape_[1] = nInputs_;
    outputShape_[0] = 1;
    outputShape_[1] = nOutputs_;
    
    if (verbose)
      std::cout << "nInputs: " << nInputs_ << ", nOuputs: " << nOutputs_ << std::endl;
  }
  OnnxDNNEvaluator(const OnnxDNNEvaluator &) = delete;
  OnnxDNNEvaluator(OnnxDNNEvaluator &&) = delete;
  OnnxDNNEvaluator &operator=(const OnnxDNNEvaluator &) = delete;
  OnnxDNNEvaluator &operator=(OnnxDNNEvaluator &&) = delete;

  ROOT::RVec<float> run(ROOT::RVec<float> in) {
    assert(in.size() == nInputs_);
    ROOT::RVec<float> out(nOutputs_);
    
    auto inputTensor =
        Ort::Value::CreateTensor<float>(memoryInfo_, &in[0], in.size(), inputShape_.data(), inputShape_.size());
    auto outputTensor =
        Ort::Value::CreateTensor<float>(memoryInfo_, (taggerType_ == "hjtagger") ? &out[1] : &out[0], out.size(), outputShape_.data(), outputShape_.size());
    
    session_.Run(Ort::RunOptions{nullptr}, inputNameC_.data(), &inputTensor, 1, outputNameC_.data(), &outputTensor, 1);
    
    return out;
  }

  unsigned int nInputs() const { return nInputs_; }
  unsigned int nOutputs() const { return nOutputs_; }

protected:
  std::string modelFile_, inputName_, outputName_;
  std::string taggerType_;
  unsigned int nInputs_, nOutputs_;
  Ort::Env env_;
  Ort::SessionOptions options_;
  Ort::MemoryInfo memoryInfo_;
  Ort::Session session_;
  std::array<int64_t, 2> inputShape_, outputShape_;
  std::array<const char *, 1> inputNameC_, outputNameC_;

  static Ort::SessionOptions defaultOpts() {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    return options;
  }
};

#endif
