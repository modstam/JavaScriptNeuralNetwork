function dsigmoid(n) {
    //sigmoid prime is really sigmoid(n) * (1 - sigmoid(n))
    //but in our case we store all the activations already so 
    //we just send in those values instead resulting in n * (1 - n)
    return n * (1 - n);
}

function sigmoid(n){
    return 1/(1+(Math.exp(-n)))
}

function softplus(n){
    return Math.log(1+Math.exp(n));
}

function dsoftplus(n){
    return sigmoid(n);
}

function randomize(n) {
    return (Math.random()*2)-1;
}

function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
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
            this.w_deltas = new Array(this.layers.length);
            this.biases = new Array(this.layers.length);
            this.b_deltas = new Array(this.layers.length);
            this.activations = new Array(this.layers.length);
            this.learningRate = 0.1;

            //create random weights and biases for connections between each layer (i, i + 1)
            for (var i = 1; i < this.layers.length; i++) {
                this.weights[i] = Matrix.zero(this.layers[i], this.layers[i - 1]);
                this.weights[i] = this.weights[i].eleMap(randomize);
                this.biases[i] = Matrix.zero(this.layers[i], 1);
                this.biases[i] = this.biases[i].eleMap(randomize);
                this.w_deltas[i] = Matrix.zero(this.layers[i], this.layers[i - 1]);
                this.b_deltas[i] = Matrix.zero(this.layers[i], 1);
            }
        }
    }

    feedForward(inputs) {
        this.activations[0] = new Matrix(inputs).trans();
        for (var i = 1; i < this.layers.length; i++) {
            var activation = this.activations[i - 1];
            var layerValues = this.weights[i].dot(activation);
            layerValues = layerValues.plus(this.biases[i]);
            layerValues = layerValues.eleMap(sigmoid);
            this.activations[i] = layerValues;

        }
        return this.activations[this.activations.length - 1];
    }

    backPropagate(outputs, targets) {
        var numL = this.layers.length;
        var input_targets = new Matrix(targets);

        var l_error = outputs.minus(input_targets).eleMap(dsigmoid);
        this.b_deltas[numL-1] = l_error;
        this.w_deltas[numL-1] = l_error.dot(this.activations[numL-2].trans());

        for (var i = numL - 2; i > 0; i--) { 
            var l_error = this.weights[i+1].trans().dot(l_error).eleMap(dsigmoid);
            this.b_deltas[i] = l_error;
            this.w_deltas[i] = l_error.dot(this.activations[i-1].trans());
        }
        // console.log(this.w_deltas);

        //update all weights, biases with deltas
        for(var i = 1; i < numL; i++){
            var w_update = this.w_deltas[i].mulEach(this.learningRate);
            var b_update = this.b_deltas[i].mulEach(this.learningRate);
            
            this.weights[i] = this.weights[i].minus(w_update);
            this.biases[i] = this.biases[i].minus(b_update);
        }                
    }

    train(inputs, targets) {     
        var result = this.feedForward(inputs);
        this.backPropagate(result, targets);
    }
}