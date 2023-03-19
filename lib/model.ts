import { Board, CellIndex, Player, State } from "./game";
import tf from "@tensorflow/tfjs";

export class Model {
    private network: tf.LayersModel;

    constructor(network?: tf.LayersModel) {
        this.network = network ?? this.createNetwork()
    }

    private static compile(model : tf.LayersModel)
    {
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
    }

    async clone(): Promise<Model>
    {
        const modelData = new Promise<tf.io.ModelArtifacts>(resolve => this.network.save({save: resolve as any}));
        const model = await tf.loadLayersModel({load: () => modelData});

        Model.compile(model);

        return new Model(model);
    }

    private createNetwork(): tf.Sequential {
        const model: tf.Sequential = tf.sequential();

        model.add(tf.layers.inputLayer({ inputShape: [9] }))
        model.add(tf.layers.dense({ units: 10, activation: "relu", name: 'hidden1', useBias: true }));
        model.add(tf.layers.dense({ units: 25, activation: 'relu', name: 'hidden2' }));
        model.add(tf.layers.dense({ units: 9, activation: 'linear', /*useBias: true, */ name: 'output' }));

        // model.summary();
        Model.compile(model);
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

        return JSON.stringify(data, undefined, 4);
    }

    public static async fromJson(str: string|object): Promise<Model> {
        const json = typeof str === 'string' ? JSON.parse(str) : str;
        json.weightData = typeof json.weightData === 'string' ? new Uint8Array(Buffer.from(json.weightData, "base64")).buffer : json.weightData;
        const network = await tf.loadLayersModel(tf.io.fromMemory(json));
        network.summary();
        return new Model(network as tf.Sequential);
    }

    public stateAsTensor(state: State | Uint8Array | Float32Array | Int32Array): tf.Tensor2D {
        return tf.tensor2d(state, [1, 9]);
    }

    public async predict(board: Board): Promise<CellIndex[]> {
        const input = this.stateAsTensor(board.getState());
        const output = tf.tidy(() => this.network.predict(input));
        if (Array.isArray(output))
            throw Error("Single tensor expected");

        const data: Float32Array = await output.data() as Float32Array;
        const epsilon = 0.000001;
        const player = board.player;

        const greaterThan = player === Player.X ? (a, b) => a > b : (a, b) => a < b;
        const lessThan = (a,b) => !greaterThan(a,b);

        const maxData = data.reduce((acc, value, index) => {
            if (board.busy(index))
                return acc;

            const delta = value - acc.value;
            if (greaterThan(delta, epsilon))
                return { value: value, index: [index] }

            if (lessThan(Math.abs(delta), epsilon))
                acc.index.push(index);

            return acc;
        }, { value: -Infinity, index: [] as number[] });
        return maxData.index;
    }

    private async predictAsTensor(state: State): Promise<{input: tf.Tensor, output: tf.Tensor}>
    {
        const input = this.stateAsTensor(state);
        const output = this.network.predict(input);
        if (Array.isArray(output)) {
            throw new Error("Tensor expected");
        }

        return {input, output};
    }

    private async predictAsVector(state: State):  Promise<{input: tf.Tensor, output: Float32Array | Int32Array | Uint8Array}>
    {
        const result = await this.predictAsTensor(state);

        return {input: result.input, output: await result.output.data()};
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
            const {input, output: predicted} =  await this.predictAsTensor(elem.curState);
            if (first) {
                const str = Array.from(await predicted.data()).map( e => e.toFixed(3) );
                console.log(`${elem.newState}: ${str}`);
            }

            const addVectorData = Array(9).fill(0);
            addVectorData[elem.move] += reward;

            const addVector = tf.tensor1d(addVectorData);
            const rewarded = predicted.add(addVector);

            // outputArray[elem.move] += reward;
            // const output = this.stateAsTensor(outputArray);

            // normalize
            const max = rewarded.max();
            const min = rewarded.min();
            const one = tf.ones([1,1]);
            const divisor = tf.where(max.notEqual(min), max.sub(min), one);
            const scaled = rewarded.sub(min).div(divisor);

            const batchSize = 32;
            const epochs = 20;

            console.assert ((await scaled.data()).every( x => x <= 1 ));

            await this.network.fit(input, scaled, {
                batchSize,
                epochs,
            });

            if (first) {
                const {output} = await this.predictAsVector(elem.curState);
                const str = Array.from(output).map( e => e.toFixed(3) );
                console.log(`${elem.newState}: ${str}`);
                first = false;
            }
        }
    }
}
