function dsigmoid(n) {
    return n * (1 - n);
}

function randomize(n) {
    return Math.random()*2-1;
}

var linAlg = new linearAlgebra(),
    Vector = linAlg.Vector,
    Matrix = linAlg.Matrix;

class NeuralNetwork {
    constructor(layers) {
        if (!(layers instanceof Array)) {
            console.log("input should be an array with length representing number " +
                "of layers and each value the number of neurons in the layer");
        } else {
            this.layers = layers;
            this.weights = new Array(this.layers.length);
            this.biases = new Array(this.layers.length);
            this.calculatedLayerValues = new Array(this.layers.length);
            this.learningRate = 0.5;

            //create random weights and biases for connections between each layer (i, i + 1)
            for (var i = 1; i < this.layers.length; i++) {
                this.weights[i] = Matrix.zero(this.layers[i], this.layers[i - 1]);
                this.weights[i] = this.weights[i].eleMap(randomize);
                this.biases[i] = Matrix.zero(this.layers[i], 1);
                this.biases[i] = this.biases[i].eleMap(randomize);
            }
        }
    }

    feedForward(inputs) {
        this.calculatedLayerValues[0] = new Matrix(inputs).trans();
        for (var i = 1; i < this.layers.length; i++) {
            var input_array = this.calculatedLayerValues[i - 1];
            var layerValues = this.weights[i].dot(input_array);
            layerValues = layerValues.plus(this.biases[i]);
            layerValues = layerValues.sigmoid();
            this.calculatedLayerValues[i] = layerValues;

        }
        return this.calculatedLayerValues[this.calculatedLayerValues.length - 1];
    }

    backPropagate(outputs, targets) {
        //delta of each layers values should be learningRate*error*dsigmoid*transpose(calculatedLayerValue)
        var input_targets = new Matrix(targets).trans();
        var errors = input_targets.minus(outputs);

        for (var i = this.layers.length - 1; i > 0; i--) {
            var gradient = this.calculatedLayerValues[i].eleMap(dsigmoid);
            gradient = gradient.mul(errors);
            gradient = gradient.mulEach(this.learningRate);
            this.biases[i] = this.biases[i].plus(gradient);

            var prevLayerValue_T = this.calculatedLayerValues[i - 1].trans();
            var weightDelta = gradient.dot(prevLayerValue_T);
            this.weights[i] = this.weights[i].plus(weightDelta);

            var weights_T = this.weights[i].trans();

            errors = weights_T.dot(errors);
        }
    }

    train(inputs, targets) {     
        var result = this.feedForward(inputs);
        this.backPropagate(result, targets);
    }
}