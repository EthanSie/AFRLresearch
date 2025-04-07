/****************************************************
 * c4solver.cpp  – Simple C++ Connect-4 solver
 * 
 * Reads input:
 *    1) currentPlayer (1 or -1)
 *    2) 6 lines, each with 7 integers for the board:
 *       board[row][col] in {0,1,-1}, row=0 is TOP
 *
 * Prints output:
 *    best_move=X
 *
 * Usage:
 *   echo "1\n0 0 0 0 0 0 0\n..." | ./c4solver
 ****************************************************/
#include <bits/stdc++.h>
using namespace std;

static const int WIDTH = 7;
static const int HEIGHT = 6;

// We'll store the board in a 2D array board[r][c].
int boardState[HEIGHT][WIDTH]; // 0 = empty, 1 = red, -1 = yellow

// Check if a column is playable
bool canPlayCol(int col) {
    return (boardState[0][col] == 0);
}

// Drop piece in col
void dropPiece(int col, int player) {
    for(int r = HEIGHT-1; r >= 0; r--){
        if(boardState[r][col] == 0){
            boardState[r][col] = player;
            break;
        }
    }
}

// Remove top piece from col
void removePiece(int col) {
    for(int r = 0; r < HEIGHT; r++){
        if(boardState[r][col] != 0){
            boardState[r][col] = 0;
            break;
        }
    }
}

// Check if "player" has a 4 in a row
bool checkWin(int player) {
    // Horizontal
    for(int r=0; r<HEIGHT; r++){
        for(int c=0; c<WIDTH-3; c++){
            if(boardState[r][c]==player && boardState[r][c+1]==player
               && boardState[r][c+2]==player && boardState[r][c+3]==player){
                return true;
            }
        }
    }
    // Vertical
    for(int c=0; c<WIDTH; c++){
        for(int r=0; r<HEIGHT-3; r++){
            if(boardState[r][c]==player && boardState[r+1][c]==player
               && boardState[r+2][c]==player && boardState[r+3][c]==player){
                return true;
            }
        }
    }
    // Diagonal 
    for(int r=0; r<HEIGHT-3; r++){
        for(int c=0; c<WIDTH-3; c++){
            if(boardState[r][c]==player && boardState[r+1][c+1]==player
               && boardState[r+2][c+2]==player && boardState[r+3][c+3]==player){
                return true;
            }
        }
    }
    // Diagonal 
    for(int r=3; r<HEIGHT; r++){
        for(int c=0; c<WIDTH-3; c++){
            if(boardState[r][c]==player && boardState[r-1][c+1]==player
               && boardState[r-2][c+2]==player && boardState[r-3][c+3]==player){
                return true;
            }
        }
    }
    return false;
}

// Check if board is full
bool isFull() {
    for(int c=0; c<WIDTH; c++){
        if(boardState[0][c] == 0) return false;
    }
    return true;
}

// Negamax alpha-beta
int negamax(int depth, int alpha, int beta, int currentPlayer) {
    // If previous move by -currentPlayer is a winning move -> we are losing
    if(checkWin(-currentPlayer)) {
        // The deeper we are, the less negative. We can encode that as +depth or -depth.
        // We'll do a standard big negative + small offset for depth:
        return -10000 + depth;
    }
    if(isFull()) {
        return 0; // draw
    }
    int value = -999999;
    
    // Column order: center first
    static int colOrder[7] = {3,2,4,1,5,0,6};
    for(int i=0; i<7; i++){
        int col = colOrder[i];
        if(canPlayCol(col)){
            dropPiece(col, currentPlayer);
            int v = -negamax(depth+1, -beta, -alpha, -currentPlayer);
            removePiece(col);
            if(v > value){
                value = v;
            }
            if(value > alpha){
                alpha = value;
            }
            if(alpha >= beta){
                break; // prune
            }
        }
    }
    return value;
}

// Return best move
int getBestMove(int currentPlayer){
    int bestMove = -1;
    int bestVal = -999999;
    int alpha = -999999;
    int beta = 999999;

    static int colOrder[7] = {3,2,4,1,5,0,6};
    for(int i=0; i<7; i++){
        int col = colOrder[i];
        if(canPlayCol(col)){
            dropPiece(col, currentPlayer);
            int val = -negamax(0, -beta, -alpha, -currentPlayer);
            removePiece(col);
            if(val > bestVal){
                bestVal = val;
                bestMove = col;
            }
            if(val > alpha){
                alpha = val;
            }
        }
    }
    return bestMove;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Input format:
    // 1) integer currentPlayer (1 or -1)
    // 2) 6 lines each with 7 integers in {0,1,-1}
    int currentPlayer;
    cin >> currentPlayer;
    for(int r=0; r<HEIGHT; r++){
        for(int c=0; c<WIDTH; c++){
            cin >> boardState[r][c];
        }
    }
    // solve
    int best = getBestMove(currentPlayer);
    cout << "best_move=" << best << "\n";
    return 0;
}
