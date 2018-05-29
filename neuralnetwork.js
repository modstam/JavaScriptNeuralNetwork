class NeuralNetwork {
    constructor(layers) {
        if (!(layers instanceof Array)) {
            console.log("input should be an array with length representing number " +
                "of layers and each value the number of neurons in the layer");
        } else {
            this.weights = new Array(layers.length, layers.length);
            this.biases = new Array(layers.length, layers.length);
            for (var i = 0; i < layers.length - 1; i++) {
                var size = [layers[i + 1], layers[i]];
                console.log(size);
                this.weights[i, i + 1] = Matrix.randomize(size, 0, 1);
                this.biases[i, i + 1] = Matrix.randomize([layers[i + 1]], 0, 1);
            }
        }
    }

    sigmoid(n) {

    }

    feedForward() {

    }

    backPropagate() {

    }

    train() {

    }
}