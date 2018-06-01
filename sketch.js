var nn;
var maxEpochs = 300;

//XOR
var training_data = [
    {
        inputs: [0, 0],
        targets: [0]
    },
    {
        inputs: [1, 0],
        targets: [1]
    },
    {
        inputs: [0, 1],
        targets: [1]
    },
    {
        inputs: [1, 1],
        targets: [0]
    },
];

// var training_data = [
//     {
//         inputs: [1, 1],
//         targets: [0]
//     },
// ];

function setup() {
    nn = new NeuralNetwork([2, 4, 1]);

    for (var i = 0; i < maxEpochs; i++) {
        var data = random(training_data);
        //console.log(data.inputs);
        nn.train(data.inputs, data.targets);
    }

    console.log(nn);
    for(var i = 0; i < training_data.length; i++){
        var result = nn.feedForward(training_data[i].inputs);
        console.log("input [" + training_data[i].inputs[0] + "," + training_data[i].inputs[1] + "]: " + result.data[0])
    }
}

function draw() {
}