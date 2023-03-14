import * as tf from "@tensorflow/tfjs-node"
import { writeFileSync } from "fs";
import { Board, Player } from "../lib/game";
import { Model } from "../lib/model"

// var tf = require('@tensorflow/tfjs')
// const ai = require('./ai');
// const util = require('util')

async function main() {

    // const data = await model.asJson();

    // writeFileSync("data.json", data);

    // const model1 = Model.fromJson(data);
}

async function train() {
    const model = new Model;
    const board = new Board;

    let x_win = 0;
    let o_win = 0;
    let games = 0;

    while (true) {
        board.start();

        const players = {
            [Player.X]: () =>  model.predict(board),
            [Player.O]: () => [board.random()]
            // [Player.O]: () => board.minimax()
        };

        while (!board.complete) {

            const xMoves = players[board.player]();
            const xMove = xMoves[Math.floor(Math.random() * xMoves.length)];
            board.move(xMove);
            // board.print();
        }

        if (board.winner === Player.X) {
            x_win++;
        }
        else if (board.winner === Player.O) {
            o_win++;
        }

        games++;
        const wprob = (x_win / games);
        const sprob = 1 - (o_win / games);
        const lprob = o_win / games;

        console.log(`${games}: win=${wprob.toFixed(3)} loose=${lprob.toFixed(3)}  non-loose=${sprob.toFixed(3)}`);
        await model.train(board);
    }

}
// model.

// console.log(typeof tf);
// console.log(tf.version.tfjs);

// const network = ai.createNetwork(tf);
// const board = new Board();

// console.log('net: ', util.inspect(network));
// console.log(ai.stateAsTensor(board));


await train();
