import math

# #################### PRIVATE PROPERTY NO TOUCHING ##############################

def __is_prime(prime_list, num):
    if not prime_list:
        return True
    for i in prime_list:
        if num % i == 0:
            return False
        if i > math.sqrt(num):
            return True

def __primes_below_gen(num):
    """
    Generator for primes below num. Could sometimes be more efficient than 
    finding all primes at once.
    """
    res = []
    for i in range(2, num):
        if __is_prime(res, i):
            res.append(i)
            yield i

def __diag_from_bottom_left(grid):
    """
    Generator yielding top-left to bottom-right directed diagonals of a 2D array, 
    starting from the bottom-left corner element.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    for row_idx in range(rows - 1, -1, -1):
        yield [grid[row_idx + col][col] for col in range(min(cols, rows - row_idx))]

    for col_idx in range(1, cols):
        yield [grid[row][col_idx + row] for row in range(min(rows, cols - col_idx))]

def __diag_from_top_left(grid):
    """
    Generator yielding bottom-left to top-right directed diagonals of a 2D array, 
    starting from the top-left corner element.
    """
    rows = len(grid)
    cols = len(grid[0])

    for col_idx in range(cols):
        yield [grid[row][col_idx - row] for row in range(min(rows - 1, col_idx), -1, -1)]

    for row_idx in range(1, rows):
        yield [grid[row_idx + offset][cols - offset - 1] for offset in range(min(cols - 1, rows - 1 - row_idx), -1, -1)]

def __factorial(n):
    return prod(i for i in range(1, n + 1))

def __num_grid_paths(x, y):
    """
    Return the number of paths without backtracking from the top-left corner to the
    bottom-right corner of an x by y grid.
    """
    # TODO: somewhat suboptimal, we don't need the full factorials
    return __factorial(x + y) // (__factorial(y) * __factorial(x))

# ############################## PUBLIC TOUCHY TOUCHY ##############################

def fibonacci(num):
    """
    Generates the first num Fibonacci numbers
    """
    pp = 0
    p = 1
    for i in range(num):
        yield pp + p
        newp = pp + p
        pp = p
        p = newp


def primes_num_gen(num):
    """
    Generator for num primes.
    """
    res = []
    generated = 0
    i = 2
    while generated < num:
        if __is_prime(res, i):
            res.append(i)
            yield i
            generated += 1
        i += 1

def eratosthenes(num):
    """
    Sieve of Eratosthenes implementation. Finds all primes below num.
    """
    state = [True for _  in range(num)]
    state[0:2] = [False, False]
    for i in range(2, int(math.sqrt(num) + 1)):
        for j in range(i**2, num, i):
            state[j] = False
    return [i for i in range(num) if state[i]]

def factorize(num):
    """
    Returns a dict() of prime factors and corresponding powers for num.
    """
    factors = {}
    for divisor in __primes_below_gen(num):
        if num % divisor == 0:
            factors[divisor] = 0
            while num % divisor == 0 and num > 1:
                num = num // divisor
                factors[divisor] += 1
        if not num > 1:
            break
    return factors

def is_palindrome(num):
    """
    Checks whether num is a palindrome.
    """
    digits = list(str(num))
    split_idx = len(digits) // 2
    front = digits[:split_idx]
    back = digits[-split_idx:]
    back.reverse()
    return front == back

def prod(nums):
    """
    Returns the product of numbers in the given list.
    """
    res = 1
    for i in nums:
        res *= int(i)
    return res

def greatest_product(elem_list, num_elems):
    """
    Returns the greatest product of num_elems consecutive numbers in the elem_list.
    """
    res = 0
    for i in range(len(elem_list) - num_elems + 1):
        res = max(res, prod(elem_list[i:i + num_elems]))
    return res

def greatest_product_grid(grid, num_adjacent):
    """
    Returns the greatest product of num_adjacent factors in a grid, considering verticals,
    horizontals and diagonals.
    """
    # rowwise max prod
    res = max([greatest_product(row, num_adjacent) for row in grid])

    # colwise product
    res = max(res, max([greatest_product([row[j] for row in grid], num_adjacent) for j in range(len(grid[0]))]))

    # bottom-left to top-right diagonals
    res = max(res, max([greatest_product(i, num_adjacent) for i in __diag_from_bottom_left(grid) if len(i) >= num_adjacent]))

    # top-left to bottom-right diagonals
    res = max(res, max([greatest_product(i, num_adjacent) for i in __diag_from_top_left(grid) if len(i) >= num_adjacent]))

    return res

def triangle_num(n):
    """
    Returns the n-th triangle number.
    """
    return (n * (n + 1)) // 2

def num_divisors(n):
    """
    The Divisor function, 'nuff said.
    """
    factors = factorize(n)
    divisors = 0
    for k in sorted(factors):
        v = factors[k]
        divisors += v + divisors * v
        
    return divisors

def collatz(n):
    res = []
    while n > 1:
        res.append(n)
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    res.append(1)
    return res

# ######################################## UNTESTED ########################################

## {{{ http://code.activestate.com/recipes/577821/ (r1)
def isqrt(x):
    if x < 0:
        raise ValueError('square root not defined for negative numbers')
    n = int(x)
    if n == 0:
        return 0
    a, b = divmod(n.bit_length(), 2)
    x = 2**(a+b)
    while True:
        y = (x + n//x)//2
        if y >= x:
            return x
        x = y
## end of http://code.activestate.com/recipes/577821/ }}}

def divisors(num):
    return [i for i in range(1, num + 1) if num % i == 0]

def divisors_of_triangle(num):

    divs = set()
    if num % 2:
        odd = num
        even = num + 1
    else:
        odd = num + 1
        even = num

    for i in divisors(even // 2):
        divs.add(i)
        for j in divisors(odd):
            divs.add(j)
            divs.add(i * j)
    return divs

# ######################################## SOLUTIONS ########################################

def problem_1():
    """
    If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. 
    The sum of these multiples is 23.
    Find the sum of all the multiples of 3 or 5 below 1000.
    """
    return sum([i for i in range(1000) if i % 3 == 0 or i % 5 == 0])

def problem_2():
    """
    Each new term in the Fibonacci sequence is generated by adding the previous two terms. 
    By starting with 1 and 2, the first 10 terms will be:
    1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...

    By considering the terms in the Fibonacci sequence whose values do not exceed four million, 
    find the sum of the even-valued terms.
    """
    return sum([i for i in fibonacci(50) if i < 4*10**6 and i % 2 == 0])

def problem_3():
    """
    The prime factors of 13195 are 5, 7, 13 and 29.
    What is the largest prime factor of the number 600851475143?
    """
    return max(factorize(600851475143).keys())

def problem_4():
    """
    A palindromic number reads the same both ways. The largest palindrome made from the product
    of two 2-digit numbers is 9009 = 91  99.
    Find the largest palindrome made from the product of two 3-digit numbers.
    """
    first, second = 0, 0
    for i in range(999, 99, -1):
        for j in range(999, 99, -1):
            if is_palindrome(i * j) and i * j > first * second:
                first, second = i, j
    return ((first, second), first * second)

def problem_5():
    """
    2520 is the smallest number that can be divided by each of the numbers from 1 to 10 
    without any remainder.
    What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
    """
    # Basically, take all the numbers in the range, and starting from the end, 
    # throw out everything that can be factorized into distinct smaller numbers.
    # Squares are a special case, since you need TWO of those smaller numbers, therefore
    # when a square is met, throw out the smaller number and consider the square as
    # two of those smaller numbers (a good example is leaving 16 and throwing out {2, 4, 8}
    return prod([5, 7, 9, 11, 13, 16, 17, 19])

def problem_6():
    """
    The sum of the squares of the first ten natural numbers is,
    12 + 22 + ... + 102 = 385
    
    The square of the sum of the first ten natural numbers is,
    (1 + 2 + ... + 10)2 = 552 = 3025

    Hence the difference between the sum of the squares of the first ten natural numbers 
    and the square of the sum is 3025  385 = 2640.

    Find the difference between the sum of the squares of the first one hundred natural 
    numbers and the square of the sum.
    """
    return abs(sum(i**2 for i in range(101)) -  sum(i for i in range(101))**2)

def problem_7():
    """
    By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 
    6th prime is 13.
    What is the 10 001st prime number?
    """
    return [i for i in primes_num_gen(10000)][-1]

def problem_8(filename):
    """
    Find the greatest product of five consecutive digits in the 1000-digit number given in
    the file.
    """
    with open(filename, "r") as f:
        # cast to int() to avoid newline, back to str() to split into elements
        return greatest_product(str(int(f.readline())), 5)

def problem_9():
    """
    A Pythagorean triplet is a set of three natural numbers, a  b  c, for which,
    a² + b² = c²
    For example, 32 + 42 = 9 + 16 = 25 = 52.

    There exists exactly one Pythagorean triplet for which a + b + c = 1000.
    Find the product abc.
    """
    for a in range(1, 1000):
        for b in range(1, 1000 - a):
            c = 1000 - a - b
            if a**2 + b**2 == c**2:
                return a * b * c
    return 0

def problem_10():
    """
    The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
    Find the sum of all the primes below two million.
    """
    return sum(eratosthenes(2*10**6))

def problem_11(filename):
    """
    In the 2020 grid below, four numbers along a diagonal line have been marked in red.

    [..] <- imagine grid here

    The product of these numbers is 26  63  78  14 = 1788696.

    What is the greatest product of four adjacent numbers in any direction
    (up, down, left, right, or diagonally) in the 2020 grid?
    """
    grid = []
    for line in open(filename, "r"):
        grid.append([int(i) for i in line.split()])

    return greatest_product_grid(grid, 4)

def problem_12():
    """
    The sequence of triangle numbers is generated by adding the natural numbers. 
    So the 7th triangle number would be 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28. The first ten terms would be:
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...
    
    Let us list the factors of the first seven triangle numbers:
    
    1: 1
    3: 1,3
    6: 1,2,3,6
    10: 1,2,5,10
    15: 1,3,5,15
    21: 1,3,7,21
    28: 1,2,4,7,14,28

    We can see that 28 is the first triangle number to have over five divisors.

    What is the value of the first triangle number to have over five hundred divisors?
    """
    # TODO: reuse the prime list used in the factorization here
    # for decent performance, but for now, fuck it, this works somewhat reasonably in
    # under 2 mins. With the prime list reuse should be a few seconds.
    i = 1
    while (num_divisors(triangle_num(i)) <= 500):
        i += 1
    
    return (i, triangle_num(i))

def problem_13(filename):
    """
    Work out the first ten digits of the sum of the one-hundred 50-digit numbers in the
    given file.
    """
    sum = 0
    for line in open(filename, "r"):
        sum += int(line)
    
    return str(sum)[:10]

def problem_14(): 
    """
    The following iterative sequence is defined for the set of positive integers:

    n -> n/2 (n is even)
    n -> 3n + 1 (n is odd)

    Using the rule above and starting with 13, we generate the following sequence:

    13  40  20  10  5  16  8  4  2  1
    It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. 
    Although it has not been proved yet (Collatz Problem), it is thought that all starting 
    numbers finish at 1.

    Which starting number, under one million, produces the longest chain?

    NOTE: Once the chain starts the terms are allowed to go above one million.
    """
    # TODO: takes a while. Since it's a sequence, it overlaps, that can easily be reused
    collatz_lens = [len(collatz(i)) for i in range(1000000)]
    return collatz_lens.index(max(collatz_lens))

def problem_15():
    """
    Starting in the top left corner of a 2x2 grid, there are 6 routes (without backtracking) 
    to the bottom right corner.

    How many routes are there through a 20x20 grid?
    """
    return __num_grid_paths(20, 20)

def problem_16():
    """
    2^15 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.
    
    What is the sum of the digits of the number 2^1000?
    """
    return sum(int(i) for i in str(2**1000))

def problem_17():
    """
    If the numbers 1 to 5 are written out in words: one, two, three, four, five, then 
    there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.

    If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, 
    how many letters would be used?


    NOTE: Do not count spaces or hyphens. For example, 342 (three hundred and forty-two) 
    contains 23 letters and 115 (one hundred and fifteen) contains 20 letters. The use 
    of "and" when writing out numbers is in compliance with British usage.
    """
    raise AttributeError("IMPLEMENT ME!")
