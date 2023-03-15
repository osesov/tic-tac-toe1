// import * as tf from "@tensorflow/tfjs-node"
import * as tf from "@tensorflow/tfjs"
import * as crypto from "crypto";
import * as fs from "fs";
import { Board, Player } from "../lib/game";
import { Model } from "../lib/model"

// var tf = require('@tensorflow/tfjs')
// const ai = require('./ai');
// const util = require('util')

async function train() {
    const model = new Model;
    const board = new Board;

    let x_win = 0;
    let o_win = 0;
    let draw = 0;
    let games = 0;
    const uuid = crypto.randomUUID();
    const backupFile = `${uuid}-backup.model.json`;
    const currentFile = `${uuid}.model.json`;
    const resetTime = 60000;
    const reportTime = 5000;
    let now = Date.now();
    let nextResetTime = now + resetTime;
    let nextReportTime = now + reportTime;

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

        games++;
        if (board.winner === Player.X) {
            x_win++;
        }
        else if (board.winner === Player.O) {
            o_win++;
        }
        else {
            draw++;
        }

        await model.train(board);

        const data = await model.asJson();

        if (fs.existsSync(backupFile))
            fs.unlinkSync(backupFile);

        if (fs.existsSync(currentFile))
            fs.renameSync(currentFile, backupFile);

        fs.writeFileSync(currentFile, data);

        const now = Date.now();
        if (now > nextReportTime) {
            const wprob = (x_win / games);
            const sprob = 1 - (o_win / games);
            const lprob = o_win / games;
            const dprob = draw / games;

            console.log(`${uuid} ${games}: win=${wprob.toFixed(3)} loose=${lprob.toFixed(3)} draw=${dprob.toFixed(3)} non-loose=${sprob.toFixed(3)}`);

            nextReportTime = now + reportTime;
        }

        if (now > nextResetTime) {
            x_win = 0;
            o_win = 0;
            games = 0;

            nextResetTime = now + resetTime;
        }
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
