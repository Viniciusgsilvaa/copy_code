from typing import Callable


def bisection(function: Callable[[float], float], a: float, b: float) -> float:
    """
    finds where function becomes 0 in [a,b] using bolzano
    >>> bisection(lambda x: x ** 3 - 1, -5, 5)
    1.0000000149011612
    >>> bisection(lambda x: x ** 3 - 1, 2, 1000)
    Traceback (most recent call last):
    ...
    ValueError: could not find root in given interval.
    >>> bisection(lambda x: x ** 2 - 4 * x + 3, 0, 2)
    1.0
    >>> bisection(lambda x: x ** 2 - 4 * x + 3, 2, 4)
    3.0
    >>> bisection(lambda x: x ** 2 - 4 * x + 3, 4, 1000)
    Traceback (most recent call last):
    ...
    ValueError: could not find root in given interval.
    """
    start: float = a
    end: float = b
    if function(a) == 0:  # one of the a or b is a root for the function
        return a
    elif function(b) == 0:
        return b
    elif (
        function(a) * function(b) > 0
    ):  # if none of these are root and they are both positive or negative,
        # then this algorithm can't find the root
        raise ValueError("could not find root in given interval.")
    else:
        mid: float = start + (end - start) / 2.0
        while abs(start - mid) > 10 ** -7:  # until precisely equals to 10^-7
            if function(mid) == 0:
                return mid
            elif function(mid) * function(start) < 0:
                end = mid
            else:
                start = mid
            mid = start + (end - start) / 2.0
        return mid


def f(x: float) -> float:
    return x ** 3 - 2 * x - 5


if __name__ == "__main__":
    print(bisection(f, 1, 4000))

    import doctest

    doctest.testmod()
# Other  random code in github

"""
        In this problem, we want to determine all possible combinations of k
        numbers out of 1 ... n. We use backtracking to solve this problem.
        Time complexity: O(C(n,k)) which is O(n choose k) = O((n!/(k! * (n - k)!)))
"""
from typing import List

def generate_all_combinations(n: int, k: int) -> List[List[int]]:
    """
    >>> generate_all_combinations(n=4, k=2)
    [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    """

    result: List[List[int]] = []
    create_all_state(1, n, k, [], result)
    return result

def create_all_state(
    increment: int,
    total_number: int,
    level: int,
    current_list: List[int],
    total_list: List[List[int]]
) -> None:
    if level == 0:
        total_list.append(current_list[:])
        return
    
    for i in range(increment, total_number - level + 2):
        current_list.append(i)
        create_all_state(i + 1, total_number, level - 1, current_list, total_list)
        current_list.pop()
    

def print_all_state(total_list: List[List[int]]) -> None:
    for i in total_list:
        print(*i)

if __name__ == "__main__":
    n = 4
    k = 2
    total_list = generate_all_combinations(n, k)
    print_all_state(total_list)


from typing import List, Tuple


def get_valid_pos(position: Tuple[int, int], n: int) -> List[Tuple[int, int]]:
    """
    Find all the valid positions a knight can move to from the current position.
    >>> get_valid_pos((1, 3), 4)
    [(2, 1), (0, 1), (3, 2)]
    """

    y, x = position
    positions = [
        (y + 1, x + 2),
        (y - 1, x + 2),
        (y + 1, x - 2),
        (y - 1, x - 2),
        (y + 2, x + 1),
        (y + 2, x - 1),
        (y - 2, x + 1),
        (y - 2, x - 1),
    ]
    permissible_positions = []

    for position in positions:
        y_test, x_test = position
        if 0 <= y_test < n and 0 <= x_test < n:
            permissible_positions.append(position)

    return permissible_positions


def is_complete(board: List[List[int]]) -> bool:
    """
    Check if the board (matrix) has been completely filled with non-zero values.
    >>> is_complete([[1]])
    True
    >>> is_complete([[1, 2], [3, 0]])
    False
    """

    return not any(elem == 0 for row in board for elem in row)


def open_knight_tour_helper(
    board: List[List[int]], pos: Tuple[int, int], curr: int
) -> bool:
    """
    Helper function to solve knight tour problem.
    """

    if is_complete(board):
        return True

    for position in get_valid_pos(pos, len(board)):
        y, x = position

        if board[y][x] == 0:
            board[y][x] = curr + 1
            if open_knight_tour_helper(board, position, curr + 1):
                return True
            board[y][x] = 0

    return False


def open_knight_tour(n: int) -> List[List[int]]:
    """
    Find the solution for the knight tour problem for a board of size n. Raises
    ValueError if the tour cannot be performed for the given size.
    >>> open_knight_tour(1)
    [[1]]
    >>> open_knight_tour(2)
    Traceback (most recent call last):
    ...
    ValueError: Open Knight Tour cannot be performed on a board of size 2
    """

    board = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(n):
            board[i][j] = 1
            if open_knight_tour_helper(board, (i, j), 1):
                return board
            board[i][j] = 0

    raise ValueError(f"Open Knight Tour cannot be performed on a board of size {n}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    