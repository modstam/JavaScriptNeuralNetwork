var nn;
var maxEpochs = 100;
var learningRate = 0.5;


var bestWeightsNumEpochs;
var bestNetwork;
var bestWeightsAccuracySum = Number.MAX_SAFE_INTEGER;


function setup() {
    nn = new NeuralNetwork([2, 4, 1]);

    var training_data = training_data_xor;

    for (var i = 0; i < maxEpochs; i++) {
        var data = random(training_data);
        //console.log(data.inputs);
        nn.train(data.inputs, data.targets, learningRate);
        evaluateCurrentAccuracy(nn, training_data, i);
    }

    //console.log(nn);
    for (var i = 0; i < training_data.length; i++) {
        nn = bestNetwork;
        var result = nn.feedForward(training_data[i].inputs);
        console.log("input [" + training_data[i].inputs[0] + "," + training_data[i].inputs[1] + "]: " + result.data[0])
    }
}

function evaluateCurrentAccuracy(nn, training_data, currentEpoch) {
    var sum = 0.0;
    //console.log("Evauluating epoch " + currentEpoch + ".." );
    for (var i = 0; i < training_data.length; i++) {
        var result = nn.feedForward(training_data[i].inputs);
        var error = training_data[i].targets[0] - result.data[0];
        //console.log("error: " + error + " ("  + training_data[i].targets[0] + " - " + result.data[0] + ")")
        sum += Math.abs(training_data[i].targets[0] - result.data[0]);
        //console.log(sum);
    }
    if (bestNetwork != undefined) {
        if (bestWeightsAccuracySum > sum) {
            bestNetwork = nn.clone();
            bestWeightsAccuracySum = sum;
            bestWeightsNumEpochs = currentEpoch;
            console.log("AccuracySum: " + sum + ", Epoch: " + currentEpoch)
        }
    }
    else {
        bestNetwork = nn.weights;
        bestWeightsNumEpochs = currentEpoch;
    }
}

function draw() {
}