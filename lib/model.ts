import { Board, CellIndex, Player, State } from "./game";
import tf from "@tensorflow/tfjs";

export class Model {
    private network: tf.Sequential;

    constructor(network?: tf.Sequential) {
        this.network = network ?? this.createNetwork()
    }

    private createNetwork(): tf.Sequential {
        const model: tf.Sequential = tf.sequential();

        model.add(tf.layers.inputLayer({ inputShape: [9] }))
        model.add(tf.layers.dense({ units: 10, activation: "relu", name: 'hidden1', useBias: true }));
        model.add(tf.layers.dense({ units: 25, activation: 'relu', name: 'hidden2' }));
        model.add(tf.layers.dense({ units: 9, activation: 'linear', /*useBias: true, */ name: 'output' }));

        model.summary();
        model.compile({
            // sin
            loss: tf.losses.meanSquaredError,
            optimizer: tf.train.adam(),
            // metrics: ['mse'],
            //
            // loss: tf.losses.meanSquaredError,
            // optimizer: 'sgd',
            // metrics: ['accuracy'],
        });

        return model;
    }

    public async asJson(): Promise<string> {
        let result = null as tf.io.ModelArtifacts | null;

        const handler = tf.io.withSaveHandler(async (modelData) => {
            result = modelData;
            return {
                modelArtifactsInfo: {
                    dateSaved: new Date(),
                    modelTopologyType: 'JSON'
                },
            };
        });
        let x = await this.network.save(handler);
        if (!result)
            throw Error("No result");

        let data: { [k: string]: any } = result
        data.weightData = result.weightData ? Buffer.from(result.weightData).toString("base64") : "";

        return JSON.stringify(data);
    }

    public static async fromJson(str: string): Promise<Model> {
        const json = JSON.parse(str);
        json.weightData = new Uint8Array(Buffer.from(json.weightData, "base64")).buffer;
        const network = await tf.loadLayersModel(tf.io.fromMemory(json));
        network.summary();
        return new Model(network as tf.Sequential);
    }

    public stateAsTensor(state: State | Uint8Array | Float32Array | Int32Array): tf.Tensor2D {
        return tf.tensor2d(state, [1, 9]);
    }

    public predict(board: Board): CellIndex[] {
        const input = this.stateAsTensor(board.getState());
        const output = tf.tidy(() => this.network.predict(input));
        if (Array.isArray(output))
            throw Error("Single tensor expected");

        const data: Float32Array = output.dataSync() as Float32Array;
        const maxData = data.reduce((acc, value, index) => {
            if (board.busy(index))
                return acc;

            if (value > acc.value)
                return { value: value, index: [index] }

            if (value == acc.value)
                acc.index.push(index);

            return acc;
        }, { value: -Infinity, index: [] as number[] });
        return maxData.index;
    }

    private async doPredict(state: State): Promise<{input: tf.Tensor, output: Float32Array | Int32Array | Uint8Array}>
    {
        const input = this.stateAsTensor(state);
        const outputTensor = this.network.predict(input);
        if (Array.isArray(outputTensor)) {
            throw new Error("Tesor expected");
        }

        return {input, output: await outputTensor.data()};
    }

    public async train(board: Board): Promise<void> {
        if (!board.complete)
            throw new Error("Incomplete game");

        const winner = board.winner;
        const reward = winner === Player.X ? +1.0
            : winner === Player.O ? -1.0
            : 0.5

        let first = false;

        for (const elem of board.stateHistory) {
            const {input, output: outputArray} =  await this.doPredict(elem.curState);
            if (first) {
                const str = Array.from(outputArray).map( e => e.toFixed(3) );
                console.log(`${elem.newState}: ${str}`);
            }

            outputArray[elem.move] += reward;
            const output = this.stateAsTensor(outputArray);

            // normalize
            const max = output.max();
            const min = output.min();
            if (max.notEqual(min))
                output.sub(min).div(max.sub(min));

            const batchSize = 32;
            const epochs = 20;

            await this.network.fit(input, output, {
                batchSize,
                epochs,
            });
            if (first) {
                const {output} =  await this.doPredict(elem.curState);
                const str = Array.from(output).map( e => e.toFixed(3) );
                console.log(`${elem.newState}: ${str}`);
                first = false;
            }
        }
    }
}
