class Matrix {
    constructor(rows, cols) {
        console.log(rows + " , " + cols);
        this.data = math.zeros(rows, cols);
    }

    static randomize(size, min, max) {
        return math.random(size, min, max);
    }

    static map(matrix, func) {
        var matrix = math.matrix(matrix);
        return matrix.map(func)
    }

    static rowVectorTranspose(rowVector){
        //mathjs doesnt support vector transposing correctly
        //so we need to implement our own
        //this function turns a row vector into a column vector
        var rowMatrix = rowVector.toArray();
        var result = math.zeros(rowMatrix.length, 1)
        for(var i = 0; i < rowVector; i++){
            result[i] = [rowVector[i]];
        }
        return result;
    }
}