//
// Created by loveh on 25. 1. 11.
//

#ifndef ALEXNET_POOLINGLAYER_H
#define ALEXNET_POOLINGLAYER_H

#include "Layer.h"
#include "Tensor.h"

class MaxPoolingLayer : public Layer {
private:
    int poolSize;
    int stride;
    Tensor inputCache;
    Tensor maxIndices;
public:
    MaxPoolingLayer(int poolSize, int stride);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    void zeroGradients() override {}
    void updateWeights(double learningRate) override {}
};


#endif //ALEXNET_POOLINGLAYER_H
