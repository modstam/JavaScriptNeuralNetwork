

function sigmoid(n) {
    return 1 / (1 + math.exp(-n))
}

function dsigmoid(n) {
    return sigmoid(n) * (1 - sigmoid(n));
}

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
            this.maxEpochs = 100;
            this.learningRate = 0.1;

            //create random weights and biases for connections between each layer (i, i + 1)
            for (var i = 1; i < this.layers.length; i++) {
                var size = [this.layers[i], this.layers[i - 1]];
                this.weights[i] = Matrix.randomize(size, 0, 1);
                this.biases[i] = Matrix.randomize([this.layers[i]], 0, 1);
                this.calculatedLayerValues[i] = math.matrix();
            }
        }
    }

    feedForward(inputs) {
        console.log("Input:" + inputs)
        this.calculatedLayerValues[0] = inputs;
        for (var i = 1; i < this.layers.length; i++) {
            var input_array = this.calculatedLayerValues[i - 1];
            // console.log("weights: " + i);
            // console.table(this.weights[i]);
            // console.log("input_array: " + i);
            // console.table(input_array);
            var layerValues = math.multiply(this.weights[i], input_array);
            layerValues = math.multiply(this.weights[i], input_array);
            layerValues = math.add(layerValues, this.biases[i]);
            layerValues = Matrix.map(layerValues, sigmoid);
            this.calculatedLayerValues[i] = layerValues;

        }
        return this.calculatedLayerValues[this.layers.length - 1];
    }

    backPropagate(outputs, targets) {
        //delta of each layers values should be learningRate*error*dsigmoid*transpose(calculatedLayerValue)
        // console.log(outputs);
        // console.log(targets);
        // console.table(math.matrix(outputs));
        // console.table(math.matrix(targets));
        var errors = math.subtract(math.matrix(targets), outputs);

        for (var i = this.layers.length - 1; i > 0; i--) {
            // console.table(errors);  
            console.log(i);         
            var gradient = Matrix.map(this.calculatedLayerValues[i], dsigmoid);
            gradient = math.multiply(gradient, errors);
            gradient = math.multiply(gradient, this.learningRate);
            this.biases[i] = math.add(this.biases[i], gradient);
            var weightDelta = math.multiply(gradient, Matrix.rowVectorTranspose(this.calculatedLayerValues[i-1]))

            console.log(this.calculatedLayerValues[i-1]);
            console.log(Matrix.rowVectorTranspose(this.calculatedLayerValues[i-1]));            

            console.log(weightDelta);
            console.log(math.matrix(this.weights[i]));
            this.weights[i] = math.add(this.weights[i], math.matrix(weightDelta));

            errors = math.multiply(math.transpose(this.weights[i]), errors);
        }
    }

    train(inputs, targets) {

        //example inputs and targets is the xor problem
        var input_array = inputs ||
            [
                [1, 0],
                [0, 1],
                [0, 0],
                [1, 1]
            ];
        var target_array = targets ||
            [
                [1],
                [1],
                [0],
                [0]
            ];

        for (var i = 0; i < this.maxEpochs; i++) {
            var randomIndex = Math.floor(Math.random() * Math.floor(input_array.length));
            console.log("FEEDFORWARD");
            var result = this.feedForward(input_array[randomIndex]);
            //console.log(result);
            console.log("BACKPROPAGATE");
            this.backPropagate(result, target_array[randomIndex]);
        }
    }
}