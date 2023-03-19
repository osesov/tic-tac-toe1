// import * as tf from "@tensorflow/tfjs-node"
import * as fs from "fs";
import { Board, Player, Winner } from "../lib/game";
import { Model } from "../lib/model"

interface PlayerState
{
    next: (board: Board) => Promise<number[]>,
    win: number
}

function pad(n: number, s: string|number)
{
    return (Array(n).fill('0').join('') + s).substr(-n);
}
class IntervalTimer
{
    private when: number;
    private interval: number;

    public constructor(interval: number, date: Date)
    {
        this.when = date.getTime() + interval;
        this.interval = interval;
    }

    public fire(n: Date): boolean
    {
        if (n.getTime() < this.when)
            return false;

        this.when = n.getTime() + this.interval;
        return true;
    }

    public toString = () => {
        return `IntervalTimer when=${new Date(this.when)}, interval=${this.interval}`;
    }
}

interface TourneurStats
{
    winner: Winner
    x: number
    o: number
    draw: number
    total: number
}

class Tourneur
{
    private players;
    private board = new Board;
    private stats: { draw: number, total: number }

    private trainListener ?: (board: Board) => void;

    public constructor(a: (board: Board) => Promise<number[]>, b: (board: Board) => Promise<number[]>)
    {
        this.players = {
            [Player.X]: { next: (board) => a(board), win: 0},
            [Player.O]: { next: (board) => b(board), win: 0},
        }

        this.reset();
    }

    public async playSingleGame(): Promise<Winner>
    {
        const board = this.board;
        const players = this.players;

        board.start();

        while (!board.complete) {
            const xMoves = await players[board.player].next(board);
            const xMove = xMoves[Math.floor(Math.random() * xMoves.length)];
            board.move(xMove);
            // board.print();
        }

        const winner = board.winner;

        if (winner !== undefined)
            players[winner].win++;
        else
            this.stats.draw++;

        this.stats.total++;

        if (this.trainListener)
            await this.trainListener(board);
        return winner;
    }

    public reset()
    {
        this.stats = {draw: 0, total: 0};
        this.players[Player.X].win = 0;
        this.players[Player.O].win = 0;
    }

    set onTrain(listener: (board: Board) => Promise<void>) {
        this.trainListener = listener;
    }

    get winner(): Winner
    {
        if (this.players[Player.X].win > this.players[Player.O].win)
            return Player.X;

        if (this.players[Player.X].win < this.players[Player.O].win)
            return Player.O;

        return undefined;
    }

    get stat(): TourneurStats
    {
        return {
            winner: this.winner,
            x: this.players[Player.X].win,
            o: this.players[Player.O].win,
            draw: this.stats.draw,
            total: this.stats.total
        };
    }

    public async playSet(continuePlaying: number | ((board: Board) => boolean) | Date): Promise<Winner>
    {
        if (typeof continuePlaying === "number") {
            let counter = continuePlaying;
            continuePlaying = () => --counter > 0;
        }

        else if (continuePlaying instanceof Date) {
            const end = continuePlaying.getTime();
            continuePlaying = () => Date.now() < end;
        }

        do {
            await this.playSingleGame();
        } while (continuePlaying(this.board));

        return this.winner;
    }

    static modelPlayer(model: Model): (board: Board) => Promise<number[]>
    {
        return (board) => model.predict(board);
    }

    static randomPlayer(): (board: Board) => Promise<number[]>
    {
        return (board) => board.random();
    }

    static minimaxPlayer(): (board: Board) => Promise<number[]>
    {
        return (board) => board.minimax();
    }
}


async function playRandomGameSet(model: Model, n: number): Promise<TourneurStats>
{
    const tourneur = new Tourneur(Tourneur.modelPlayer(model), Tourneur.randomPlayer());
    await tourneur.playSet(n);
    return tourneur.stat;
}

async function playMinimaxGameSet(model: Model, n: number): Promise<TourneurStats>
{
    const tourneur = new Tourneur(Tourneur.modelPlayer(model), Tourneur.minimaxPlayer());
    await tourneur.playSet(n);
    return tourneur.stat;
}


async function geneticTourneur(parent: Model): Promise<Model> {
    const x = await parent.clone();
    const o = await parent.clone();

    const tourneur = new Tourneur( Tourneur.modelPlayer(x), Tourneur.modelPlayer(o));

    tourneur.onTrain = async (board: Board) => {
        printStat("progress", tourneur.stat);
        await x.train(board);
        await o.train(board);
    };

    await tourneur.playSet(64);

    const winner = tourneur.winner == Player.X ? x : o;
    return winner;
}

namespace Duration
{
    export const MILLIS = 1;
    export const SECONDS = 1000 * MILLIS;
    export const MINUTES = 60 * SECONDS;
    export const HOURS = 60 * MINUTES;
    export const DAYS = 24 * HOURS;

    export function millis(n: number)  { return n * MILLIS; }
    export function seconds(n: number) { return millis(n * SECONDS); }
    export function minutes(n: number) { return millis(n * MINUTES); }
    export function hours(n: number)   { return millis(n * HOURS); }
    export function days(n: number)    { return millis(n * DAYS); }
}


async function saveModel(name: string, model: Model)
{
    const backupFile = `${name}-backup.model.json`;
    const currentFile = `${name}.model.json`;

    const data = await model.asJson();

    if (fs.existsSync(backupFile))
        fs.unlinkSync(backupFile);

    if (fs.existsSync(currentFile))
        fs.renameSync(currentFile, backupFile);

    fs.writeFileSync(currentFile, data);

    console.log("Saved as " + currentFile);
}

function printStat(name: string, stat: TourneurStats, x?: string, o?: string)
{
    const total = stat.total || 1;

    x = x ?? "x";
    o = o ?? "o"

    console.log("TourneurStats: %s: games=%s"
        + " %s={win=%s (%s), non-lost=%s}"
        + " %s={win=%s (%s), non-lost=%s}"
        + " %s=%s (%s)"
        ,
        name, stat.total,
        x, stat.x, (stat.x / total).toFixed(3), ((stat.total - stat.o) / total).toFixed(3),
        o, stat.o, (stat.o / total).toFixed(3), ((stat.total - stat.x) / total).toFixed(3),
        "draw", stat.draw, (stat.draw / total).toFixed(3),
    )
}

async function geneticTrain()
{
    let model = new Model;

    const startDate = new Date();
    const timer = new IntervalTimer(Duration.hours(48), startDate);

    const file_name = getFileName('genetic', startDate);
    do {
        model = await geneticTourneur(model);
        await saveModel(file_name, model);

        const randomStats = await playRandomGameSet(model, 100);
        const minimaxStats = await playMinimaxGameSet(model, 100);

        printStat("random", randomStats);
        printStat("minimax", minimaxStats);

    } while(!timer.fire(new Date));

    console.log("Done");
}

function getFileName(type: string, date: Date): string
{
    const file_name = `${type}`
    + '-'
    + `${pad(4, date.getFullYear())}`
    + `${pad(2, date.getMonth())}`
    + `${pad(2, date.getDate())}`
    + '-'
    + `${pad(2, date.getHours())}`
    + `${pad(2, date.getMinutes())}`
    + `${pad(2, date.getSeconds())}`
    ;

    return file_name;
}

async function randomTrain() {

    const model = new Model;
    const tourneur = new Tourneur(Tourneur.modelPlayer(model), Tourneur.randomPlayer());

    const startTime = new Date;
    const file_name = getFileName('random', startTime);
    const reportTimer = new IntervalTimer(Duration.seconds(5), startTime);
    const resetTimer = new IntervalTimer(Duration.seconds(60), startTime);

    tourneur.onTrain = async (board: Board) => {
        await model.train(board);

        await saveModel(file_name, model);

        const now = new Date;
        if (reportTimer.fire(now)) {
            printStat("randomTrain", tourneur.stat, "model", "random");
        }

        if (resetTimer.fire(now)) {
            tourneur.reset();
        }

    };

    await tourneur.playSet(Infinity);
}

// await randomTrain();
await geneticTrain();
