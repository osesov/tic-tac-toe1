import tf from "@tensorflow/tfjs";
import json from "../f1f368cb-35d1-429b-8e28-92de62153049.model.json";

interface Window
{
    tf: typeof tf,
    // tfvis: typeof tfvis
}

declare var window: Window & typeof globalThis;
import {Model} from '../lib/model';
import {Board, Player} from '../lib/game';
import { fstat } from "fs";

export async function main()
{
    console.log(tf.version.tfjs);

    const model = await Model.fromJson(json);
    const board = new Board;
    let xWinCount = 0;
    let oWinCount = 0;
    let drawCount = 0;
    let gameCount = 0;

    while (true) {
        board.start();
        console.log('####################################');

        const players = {
            [Player.X]: () => model.predict(board),
            // [Player.O]: () => board.random(),
            [Player.O]: () => board.minimax()
        };

        while (!board.complete) {

            const xMoves = await players[board.player]();
            const xMove = xMoves[Math.floor(Math.random() * xMoves.length)];
            board.move(xMove);
            board.print();
        }

        gameCount++;
        if (board.winner === Player.X) {
            xWinCount++;
        }
        else if (board.winner === Player.O) {
            oWinCount++;
        }
        else {
            drawCount++;
        }
    }
}

if (global.window)
    $(main);
else
    main();
