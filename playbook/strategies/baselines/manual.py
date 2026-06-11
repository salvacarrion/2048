"""Manual play — a human at the keyboard. Handy for debugging environments."""
from ...game.board import Move, move_name
from ..base import Strategy


class ManualStrategy(Strategy):
    name = "manual"

    def select_move(self, board, legal):
        options = ", ".join(f"{int(m)}={move_name(m)}" for m in sorted(legal))
        while True:
            raw = input(f"Move [{options}]: ").strip()
            try:
                move = Move(int(raw))
            except (ValueError, KeyError):
                print("  invalid")
                continue
            if move in legal:
                return move
            print("  not a legal move here")
