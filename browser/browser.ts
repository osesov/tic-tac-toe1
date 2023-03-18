import tf from "@tensorflow/tfjs";

interface Window
{
    tf: typeof tf,
    // tfvis: typeof tfvis
}

declare var window: Window & typeof globalThis;
import {Model} from '../lib/model';
import {Board} from '../lib/game';

export async function main()
{
    console.log(window.tf.version.tfjs);

    const model = await Model.fromJson("f1f368cb-35d1-429b-8e28-92de62153049.model.json");

    const board = new Board;
    console.log(board);

    while (true) {
        board.start();

        while (!board.complete) {
            const xMoves = board.minimax();
            const possibleMoves = model.predict(board);

            const xMove = xMoves[Math.floor(Math.random() * xMoves.length)];
            board.move(xMove);
            board.print();

            model.predict(board);
        }

        await model.train(board);
    }
}

$(main);
