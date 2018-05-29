class Matrix {
    constructor(rows, cols) {
        console.log(rows + " , " + cols);
        this.data = math.zeros(rows, cols);
    }

    static randomize(size, min, max) {
        return math.random(size, min, max);
    }
}