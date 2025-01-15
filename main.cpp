#include "Dataset.h"
#include "DataLoader.h"
#include "Model.h"
#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "LossFunc.h"
#include "PoolingLayer.h"
#include "Optimizer.h"
#include "Tensor.h"
#include <iostream>
#include <cmath>
#include <iomanip>

int main() {

    Tensor tensor({2, 3, 4}, false); // 3D Tensor

// 1. Dataset과 DataLoader 초기화
    Dataset trainDataset("C:/Users/loveh/CLionProjects/Alexnet/MNIST_data/train-images.idx3-ubyte", "C:/Users/loveh/CLionProjects/Alexnet/MNIST_data/train-labels.idx1-ubyte");
    Dataset testDataset("C:/Users/loveh/CLionProjects/Alexnet/MNIST_data/t10k-images.idx3-ubyte", "C:/Users/loveh/CLionProjects/Alexnet/MNIST_data/t10k-labels.idx1-ubyte");

    auto trainData = trainDataset.loadData();
    auto testData = testDataset.loadData();

    int batchSize = 32;

    DataLoader trainLoader(trainData, batchSize); // 배치 크기: 32
    DataLoader testLoader(testData, batchSize); // 배치 크기: 32


    // 2. 모델 초기화
    Model model;
    model.addLayer(make_shared<ConvLayer>(1, 2, 3, 1, 1)); // Conv Layer
    model.addLayer(make_shared<Relu>());                  // ReLU 활성화 함수
    model.addLayer(make_shared<MaxPoolingLayer>(2, 2));   // Max Pooling
    model.addLayer(make_shared<Flatten>());
    model.addLayer(make_shared<FullyConnectedLayer>(2 * 14 * 14, 32)); // FC
    model.addLayer(make_shared<Relu>());                               // ReLU 활성화 함수
    model.addLayer(make_shared<FullyConnectedLayer>(32, 10)); // Output FC
    model.addLayer(make_shared<Sigmoid>());                  // Sigmoid 활성화 함수


    // 3. Optimizer 및 Loss Function 초기화
    SGD optimizer(make_shared<Model>(model), 0.01); // Learning rate: 0.01
    CrossEntropyLoss lossFunction;

    // 4. Training 루프
    for (int epoch = 0; epoch < 3; ++epoch) { // 15 epochs
        int batchIndex = 0;
        double totalLoss = 0.0;
        double totalAccuracy = 0.0;
        int totalSamples = 0;

        while (trainLoader.hasNextBatch()) {
            ImageSet batch = trainLoader.getNextBatch();
            Tensor images(batch.images);
            Tensor labels(batch.labels);


            Tensor Labels = oneHot(labels, 10);

            // Forward
            Tensor predictions = model.forward(images);

            // 손실 계산
            double loss = lossFunction.computeLoss(predictions, Labels);

            // Zero Grad
            optimizer.zero_grad();

            // Backward
            Tensor gradients = lossFunction.computeGradient(predictions, Labels);
            model.backward(gradients);

            // Update
            optimizer.step();

            // Accuracy 계산
            Tensor prob = predictions.softmax(1); // {batch, 10}, predictions의 [1] 번째를 기준으로 softmax연산을 수행하겠다.
            Tensor pred = prob.argmax(1);         // dim=1
            double accuracy = pred.eq(labels).mean();


            // Get the batch size dynamically (to account for the last batch)
            int currentBatchSize = labels.getSize();

            // Accumulate metrics
            totalLoss += loss;
            totalAccuracy += accuracy;
            totalSamples += currentBatchSize;

            ++batchIndex;

            // Log every 128 batches
            if (batchIndex % 128 == 0) {
                cout << fixed << setprecision(8)
                          << "TRAIN-Iteration: " << batchIndex
                          << ", Loss: " << loss
                          << ", Accuracy: " << accuracy
                          << endl;
            }
        }

        trainLoader.reset();
    }

// 5. 테스트
    double totalAccuracy = 0.0;
    double totalLoss = 0.0;
    int totalSamples = 0;

    testLoader.reset();

    while (testLoader.hasNextBatch()) {
        ImageSet batch = testLoader.getNextBatch();
        Tensor images(batch.images);
        Tensor labels(batch.labels);

        // Forward
        Tensor predictions = model.forward(images);

        // 손실 계산
        Tensor oneHotLabels = oneHot(labels, 10);
        double loss = lossFunction.computeLoss(predictions, oneHotLabels);

        // Accuracy 계산
        Tensor prob = predictions.softmax(1);  // dim=1
        Tensor pred = prob.argmax(1);          // dim=1
        double batchAccuracy = pred.eq(labels).mean();

        // 총 샘플 수를 증가
        totalSamples += labels.getSize();
        totalLoss += loss;
        totalAccuracy += batchAccuracy;

    }

    // 평균 Loss 및 Accuracy 계산
    double averageLoss = totalLoss / totalSamples;
    double averageAccuracy = totalAccuracy / totalSamples;

    // 결과 출력
    std::cout << "TEST-Accuracy: " << averageAccuracy * 100.0 << "%, Average Loss: " << averageLoss << std::endl;


    return 0;
}