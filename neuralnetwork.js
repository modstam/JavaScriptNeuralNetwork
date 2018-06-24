function sigmoid(n) {
    return 1 / (1 + (Math.exp(-n)))
}

function dsigmoid(n) {
    //sigmoid prime is really sigmoid(n) * (1 - sigmoid(n))
    //but in our case we store all the activations already so 
    //we just send in those values instead resulting in n * (1 - n)
    return n * (1 - n);
}

function dsigmoid_true(n) {
    return sigmoid(n) * (1 - sigmoid(n));
}



function softplus(n) {
    return Math.log(1 + Math.exp(n));
}

function dsoftplus(n) {
    return sigmoid(n);
}

function randomize(n) {
    return (Math.random() * 2) - 1;
}

function randn_bm() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
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
            this.w_deltas = new Array(this.layers.length);
            this.b_deltas = new Array(this.layers.length);
            this.activations = new Array(this.layers.length);
            this.unActivatedValues = new Array(this.layers.length);

            //create random weights and biases for connections between each layer (i, i + 1)
            for (var i = 1; i < this.layers.length; i++) {
                this.weights[i] = Matrix.zero(this.layers[i], this.layers[i - 1]).eleMap(randomize);
                this.biases[i] = Matrix.zero(this.layers[i], 1).eleMap(randomize);
                this.w_deltas[i] = Matrix.zero(this.layers[i], this.layers[i - 1]);
                this.b_deltas[i] = Matrix.zero(this.layers[i], 1);
            }
        }
    }

    feedForward(inputs) {
        this.activations[0] = Matrix.reshapeFrom(inputs, inputs.length, 1);
        for (var i = 1; i < this.layers.length; i++) {
            var prevActivation = this.activations[i - 1];
            this.unActivatedValues[i] = this.weights[i].dot(prevActivation);
            this.unActivatedValues[i] = this.unActivatedValues[i].plus(this.biases[i]);
            this.activations[i] = this.unActivatedValues[i].eleMap(sigmoid);
        }
        return this.activations[this.activations.length - 1];
    }

    backPropagate(outputs, targets, learningRate) {
        var numL = this.layers.length;
        var input_targets =  Matrix.reshapeFrom(targets, targets.length, 1);

        var sigmoid_prime = this.unActivatedValues[numL-1].eleMap(dsigmoid_true);
        //console.table(sigmoid_prime);

        var l_error = outputs.minus(input_targets).mul(sigmoid_prime);
        this.b_deltas[numL - 1] = l_error;
        this.w_deltas[numL - 1] = l_error.dot(this.activations[numL - 2].trans());
        //this.w_deltas[numL - 1] = this.activations[numL - 2].dot(l_error);

        for (var i = numL - 2; i > 0; i--) {
            sigmoid_prime = this.unActivatedValues[i].eleMap(dsigmoid_true);
            var l_error = this.weights[i + 1].trans().dot(l_error).mul(sigmoid_prime);
            this.b_deltas[i] = l_error;
            this.w_deltas[i] = l_error.dot(this.activations[i - 1].trans());
            //this.w_deltas[i] = this.activations[i - 1].dot(l_error);
        }
        // console.log(this.w_deltas);

        //update all weights, biases with deltas
        for (var i = 1; i < numL; i++) {
            var w_update = this.w_deltas[i].mulEach(learningRate);
            var b_update = this.b_deltas[i].mulEach(learningRate);

            this.weights[i] = this.weights[i].plus(w_update);
            this.biases[i] = this.biases[i].plus(b_update);

            // console.table(this.weights[i].data);
        }
    }

    train(inputs, targets, learningRate) {
        var result = this.feedForward(inputs);
        this.backPropagate(result, targets, learningRate);
    }

    clone() {
        var nn = new NeuralNetwork(this.layers);
        for (var i = 1; i < this.layers.length; i++) {
            nn.weights[i] = this.weights[i].clone();
            nn.biases[i] = this.biases[i].clone();
        }
        return nn;
    }
}