export enum Player
{
    X = 1,
    O = -1
}

export enum CellStateEx
{
    E = 0
}

export enum Score
{
    WIN = +1,
    LOST = -1,
    DRAW = 0
}

export const CellState = { ...Player, ...CellStateEx };
export type CellState = Player | CellStateEx;

export type CellIndex = number; // 0|1|2|3|4|5|6|7|8
export type State = CellState[]

export type StateHistory = {
    move: CellIndex,
    state: State
}

type MiniMax = { curScore: number, nextMove: CellIndex[]};

function invScore(score: MiniMax): number
{
    return -score.curScore;
}

export class Board
{
    private state_: State = []
    private winner_: Player | undefined;
    private cellsLeft_ = 0;
    private history_: CellIndex[] = []
    private stateHistory_ : StateHistory[] = []

    public constructor()
    {
        this.start();
    }

    start(): void
    {
        // console.log("start!")
        this.state_ = Array(9).fill(CellState.E);
        this.winner_ = undefined
        this.cellsLeft_ = 9;
        this.history_ = [];
        this.stateHistory_ = [];
    }

    get complete(): boolean
    {
        return this.cellsLeft_ === 0 || this.winner_ !== undefined
    }

    get player(): Player
    {
        return this.cellsLeft_ % 2 === 1 ? Player.X : Player.O
    }

    get winner(): Player | undefined
    {
        return this.winner_
    }

    get history()
    {
        return this.history_.slice(0);
    }

    get stateHistory(): StateHistory[]
    {
        return this.stateHistory_;
    }

    private checkWinner(): void
    {
        const cross = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]

        const winner = cross.find((cells: number[]) => {
            const p0 = this.state_[cells[0]];
            const p1 = this.state_[cells[1]];
            const p2 = this.state_[cells[2]];
            return p0 != CellState.E && p0 == p1 && p0 == p2;
        });

        if (!winner)
            return

        const cell = this.state_[ winner[0] ];

        if (cell === CellState.E)
            return

        this.winner_ = cell;
    }

    move(index: CellIndex)
    {
        if (index < 0 || index > 8 || Math.floor(index) != index)
            throw new Error(`Index expected to be int in range [0..8]. got: ${index}`);
        if (this.state_[index] !== CellState.E)
            throw new Error(`Cell ${index} is occupied`);

        if (this.complete)
            return;

        this.history_.push(index);
        this.state_[index] = this.player;
        this.stateHistory_.push( { state: this.state_.slice(0), move: index})
        this.cellsLeft_ --;
        this.checkWinner();
    }

    undo()
    {
        const index = this.history_.pop();
        if (index === undefined)
            return;

        this.stateHistory_.pop();
        this.state_[index] = CellState.E;
        this.cellsLeft_++;
        this.winner_ = undefined;
    }

    public random(): number
    {
        const indexes: number[] = this.state_.reduce((result: number[], currentValue: CellState, currentIndex: number) => {
            if (currentValue === CellState.E)
                result.push(currentIndex);

            return result;
        }, [] as number[]);

        const index = Math.random() * indexes.length;
        return indexes[Math.floor(index)];
    }

    public minimax(): number[]
    {
        return this.minimaxCore().nextMove;
    }

    private minimaxCore(): MiniMax
    {
        let bestMove : CellIndex[] = [];

        if (this.complete) {
            const winner = this.winner;
            if (winner === undefined)
                return { curScore: 0, nextMove: bestMove };
            if (winner === this.player)
                return { curScore: +1, nextMove: bestMove};
            return { curScore: -1, nextMove: bestMove};
        }

        let maxScore = -Infinity;

        for (let i = 0; i < 9; ++i) {
            let index = i as CellIndex;
            if (this.state_[index] !== CellState.E)
                continue;

            this.move(i as CellIndex);
            let currScore = invScore(this.minimaxCore());
            this.undo();

            if (currScore > maxScore) {
                maxScore = currScore;
                bestMove = [index];
            } else if (currScore === maxScore) {
                bestMove.push(index);
            }
        }

        if (bestMove.length === null)
            return { curScore: 0, nextMove: bestMove };

        return {curScore: maxScore, nextMove: bestMove };
    }

    getState(): State
    {
        return this.state_.slice(0);
    }

    busy(cell: CellIndex): boolean
    {
        return this.state_[cell] != CellState.E;
    }

    cellAsString(i: CellIndex): string
    {
        switch (this.state_[i]) {
            case CellState.X: return 'x';
            case CellState.O: return 'o';
            case CellState.E: return ' ';
        }
    }

    rowAsString(i: number): string
    {
        const begin = i * 3;
        const c0 = this.cellAsString(begin + 0);
        const c1 = this.cellAsString(begin + 1);
        const c2 = this.cellAsString(begin + 2);

        return ` ${c0} | ${c1} | ${c2} `;
    }

    asString(): string
    {
        const lineSep = "\n";
        const rowSep = "-----------" + lineSep;
        const r0 = this.rowAsString(0) + lineSep;
        const r1 = this.rowAsString(1) + lineSep;
        const r2 = this.rowAsString(2) + lineSep;

        return `${r0}${rowSep}${r1}${rowSep}${r2}`;
    }

    print()
    {
        console.log(this.asString());
    }
}
