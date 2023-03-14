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

    const model = new Model(window.tf);
    console.log(model);

    const board = new Board;
    console.log(board);

    // board.move(0);
    // board.move(2);
    // board.move(6);
    // board.move(8);
    // board.move(1);
    // board.move(7);
    // board.move(4);
    // board.print()
    // let p = board.minimax();

    while (true) {
        board.start();

        while (!board.complete) {
            const xMoves = board.minimax();
            const xMove = xMoves[Math.floor(Math.random() * xMoves.length)];
            board.move(xMove);
            board.print();

            model.predict(board);
        }

        await model.train(board);
    }
}

$(main);
