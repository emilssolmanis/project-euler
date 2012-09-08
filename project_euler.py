# -*- coding: utf-8 -*-

import math, fractions

# #################### PRIVATE PROPERTY NO TOUCHING ##############################

def __is_prime(num, prime_list=None):
    if num < 2:
        return False
    if prime_list is None:
        for p in __primes_below_gen(num):
            if num % p == 0:
                return False
        return True
    else:
        if not prime_list:
            return True
        for i in prime_list:
            if i > math.sqrt(num):
                return True
            if num % i == 0:
                return False

def __primes_below_gen(num):
    """
    Generator for primes below num. Could sometimes be more efficient than 
    finding all primes at once.
    """
    res = []
    for i in range(2, num):
        if __is_prime(i, res):
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

def __num_grid_paths(x, y):
    """
    Return the number of paths without backtracking from the top-left corner to the
    bottom-right corner of an x by y grid.
    """
    # TODO: somewhat suboptimal, we don't need the full factorials
    return factorial(x + y) // (factorial(y) * factorial(x))

def __num_to_text(num):
    """
    Returns a string for num, like "two hundred and fourty seven" for 247. Works in range [1; 1000]
    """
    singles = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if num < 100:
        if num < 10:
            return singles[num]
        elif 9 < num < 20:
            return teens[num % 10]
        else:
            if num % 10 != 0:
                return "{}-{}".format(tens[num // 10], __num_to_text(num % 10))
            else:
                return "{}".format(tens[num // 10])
    elif num < 1000:
        if num % 100 != 0:
            return "{} hundred and {}".format(singles[num // 100], __num_to_text(num % 100))
        else:
            return "{} hundred".format(singles[num // 100])
    else:
        return "one thousand"

def __triangle_max_sum(triangle):
    """
    Returns the max-sum path in a triangle from the top to the bottom.

    The algorithm uses dynamic programming, it starts from the bottom and calculates the max sum
    we can get at the next-to-last row, then recurses onto the just obtained smaller triangle.
    """
    if (len(triangle) == 1):
        return triangle[0][0]

    for idx, elem in enumerate(triangle[-2]):
        triangle[-2][idx] += max(triangle[-1][idx], triangle[-1][idx + 1])

    return __triangle_max_sum(triangle[:-1])

def __is_leap_year(year):
    """
    Returns whether the year is a leap year
    """
    if year % 100 == 0:
        return year % 400 == 0
    else:
        return year % 4 == 0

def __days_in_month(month, year):
    """
    Returns the number of days in month, year.
    """
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if month == 2 and __is_leap_year(year):
        return days[month - 1] + 1
    else:
        return days[month - 1]

def __alphabetical_value(word):
    """
    Returns an alphabetical value for a given word, where A = 1, B = 2, etc.
    """
    return sum(ord(c) - ord('A') + 1 for c in word.upper())

def __as_sum(num_set, n):
    """
    Returns whether n can be expressed as a sum of 2 numbers in num_set.
    """
    for num in num_set:
        if n - num in num_set:
            return True
    return False

def __as_sum_full(num_set, n):
    """
    Returns a list of possible additives to obtain n from numbers in num_set.
    """
    res = set()
    for num in num_set:
        if n - num in num_set:
            if n - num >= num:
                res.add((num, n - num))

    return res

def __abundant_below(n):
    prime_list = eratosthenes(n)
    perfect = []
    previous_abundant = []
    for i in range(1, n):
        found = False
        for j in previous_abundant:
            if i % j == 0:
                yield i
                found = True
                break
        for j in perfect:
            if i % j == 0:
                yield i
                found = True
                break

        if found:
            continue

        s = proper_divisor_sum(i, prime_list)

        if i == s:
            perfect.append(i)
        elif i < s:
            previous_abundant.append(i)
            yield i

def __lex_permutations(elem_set):
    """
    Generates permutations in lexicographic order. Could rewrite to iterative and
    obtain generator, but recursive is more natural.
    """
    permutations = []
    if len(elem_set) == 1:
        permutations.append(list(elem_set))
    else:
        for elem in sorted(elem_set):
            for perm in __lex_permutations(elem_set.difference({elem})):
                permutations.append([elem] + perm)

    return permutations

def __combinations(elem_set, k):
    """
    Returns all possible outcomes of elem_set choose k.
    """
    if not k or not elem_set:
        return []

    res = []
    elem_list = sorted(elem_set)

    for idx, elem in enumerate(elem_list):
        if k == 1:
            res.append({elem})
        for c in __combinations(elem_list[idx+1:], k - 1):
            res.append({elem}.union(c))

    return res

def __order(a, n):
    """
    Returns the order of a modulo n.
    """
    if fractions.gcd(a, n) != 1:
        raise ValueError("GCD of a and n is not 1")

    i = 1
    while a**i % n != 1:
        i += 1
    return i

def __reciprocal_length(denominator):
    """
    Returns the length of the recurrent part in decimal notation for a fraction 1 / denominator.
    """
    while denominator % 2 == 0:
        denominator //= 2
    
    while denominator % 5 == 0:
        denominator //= 5

    # means it consisted of only 2 and 5 in the first place, it is terminal
    if denominator == 1:
        return 0
    else:
        return __order(10, denominator)

def __coin_permutations(nominals, target_sum):
    """
    Returns the possible ways to get target_sum as a sum of an arbitrary number of elements in nominals -- that is,
    the possible partitions of target_sum. If compositions are needed, change "nominals[idx:]" to nominals
    """
    permutations = []
    if target_sum <= 0:
        return []
    for idx, coin in enumerate(sorted(nominals)):
        if coin == target_sum:
            permutations.append([coin])
        for perm in __coin_permutations(nominals[idx:], target_sum - coin):
            permutations.append([coin] + perm)

    return permutations

def __is_mult(perm):
    """
    Checks whether the given list of numbers can be split into consecutive parts of factor | factor | product.
    """
    s = str().join(str(i) for i in perm)

    # maintain two indices -- one that splits off factor_1 from factor_2
    # and another one that splits factor_2 and product

    # for end - 1 to 
    for p_idx in range(len(s) - 1, 1, -1):
        for f2_idx in range(p_idx - 1, 0, -1):
            if int(s[:f2_idx]) * int(s[f2_idx:p_idx]) == int(s[p_idx:]):
                return (True, int(s[:f2_idx]), int(s[f2_idx:p_idx]), int(s[p_idx:]))

    return (False, 0, 0, 0)

def __rotations(n):
    """
    Generates circular rotations of n.
    """
    s = str(n)
    for i in range(len(s)):
        yield int(s[i:] + s[:i])

def __is_truncatable(n, prime_list=None):
    """
    A number is "truncatable" if it is prime, and it remains prime if you truncate it
    from any side. E. g., 3797, 797, 97, and 7. Similarly, right to left: 3797, 379, 
    37, and 3.
    """
    if not __is_prime(n, prime_list):
        return False

    s = str(n)
    for i in range(1, len(s)):
        if not __is_prime(int(s[i:]), prime_list) or not __is_prime(int(s[:-i]), prime_list):
            return False

    return True

def __reverse_pent_number(p):
    """
    Reverse Pentagon number calculation. A pentagon number is given by the formula
    P_n = n (3n - 1) / 2, so e.g., P_5 = 35. This function does the opposite,
    foo(35) = 5
    """
    # We don't need to consider the other root of the sqr equation, because it's always
    # negative.
    return (1 + math.sqrt(24 * p + 1)) / 6

def __reverse_hex_number(h):
    """
    Inverse hex-number calculation, a hexagonal number is H_n = n(2n-1), if
    we solve the square equation we get the inverse.
    """
    return (1 + math.sqrt(8 * h + 1)) / 4

def __is_pent_number(n):
    """
    Checks whether n is a pentagon number.
    """
    return is_int(__reverse_pent_number(n))

def __is_hex_number(n):
    """
    Checks whether n is a hexagonal number.
    """
    return is_int(__reverse_hex_number(n))

# ############################## PUBLIC TOUCHY TOUCHY ##############################

def fibonacci(num=None):
    """
    Generates the first num Fibonacci numbers if num is given, or an infinite series
    """
    pp = 0
    p = 1
    if num:
        for i in range(num):
            yield pp + p
            newp = pp + p
            pp = p
            p = newp
    else:
        while True:
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
        if __is_prime(i, res):
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

def factorize(num, prime_list=None):
    """
    Returns a dict() of prime factors and corresponding powers for num.
    """
    factors = {}
    if prime_list:
        for divisor in prime_list:
            if num % divisor == 0:
                factors[divisor] = 0
            while num % divisor == 0 and num > 1:
                num = num // divisor
                factors[divisor] += 1
            if not num > 1:
                break
    else:
        for divisor in __primes_below_gen(num):
            if num % divisor == 0:
                factors[divisor] = 0
            while num % divisor == 0 and num > 1:
                num = num // divisor
                factors[divisor] += 1
            if not num > 1:
                break
    return factors

def is_palindrome(s):
    """
    Checks whether s is a palindrome.
    """
    return not s or (s[0] == s[-1] and is_palindrome(s[1:-1]))

def factorial(n):
    """
    Returns n!
    """
    return prod(i for i in range(1, n + 1))

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

def pentagon_num(n):
    """
    Returns the n-th pentagonal number.
    """
    return n * (3 * n - 1) // 2

def num_divisors(n, prime_list=None):
    """
    The Divisor function, 'nuff said.
    """
    factors = factorize(n, prime_list)
    divisors = 0
    for k in sorted(factors):
        v = factors[k]
        divisors += v + divisors * v
        
    return divisors

def divisors(n, prime_list=None):
    """
    Returns a set of proper divisors for n (including n).
    """
    factors = factorize(n, prime_list)
    divisors = {1}
    for k in sorted(factors):
        v = factors[k]
        divs_pows = set()
        # add all the powers of k
        for i in range(v):
            divs_pows.add(k**(i + 1))

        # add all the c * powers of k
        divs_mults = set()
        for p in divs_pows:
            for d in divisors:
                divs_mults.add(p * d)

        divisors = divisors.union(divs_pows)
        divisors = divisors.union(divs_mults)
    return divisors
    
def proper_divisor_sum(n, prime_list=None):
    """
    Returns the sum of the proper divisors of n.
    """
    return sum(divisors(n, prime_list).difference({n}))

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

def is_int(num):
    """
    Rough check whether num is an integer, used for floats to check whether the fractional
    part is 0.
    """
    return num - int(num) == 0

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
            if is_palindrome(str(i * j)) and i * j > first * second:
                first, second = i, j
    return first * second

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

def problem_8(filename="problem_8.dat"):
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

def problem_11(filename="problem_11.dat"):
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
    # TODO: consider case that 10**5 might not be enough, but the generator thing
    # is just too damn slow

    primes = eratosthenes(10**5)

    i = 1
    while (num_divisors(triangle_num(i), primes) <= 500):
        i += 1
    
    return triangle_num(i)

def problem_13(filename="problem_13.dat"):
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
    seq = {1: 1}

    for n in range(2, 10**6):
        l = 0
        curr_n = n
        while curr_n > 1 and not curr_n in seq:
            l += 1
            if curr_n % 2 == 0:
                curr_n //= 2
            else:
                curr_n = 3 * curr_n + 1
            if curr_n in seq:
                seq[n] = seq[curr_n] + l

    return max(seq, key=seq.get)

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
    return sum([len(__num_to_text(i).replace(" ", "").replace("-", "")) for i in range(1, 1001)])

def problem_18(filename="problem_18.dat"):
    """
    By starting at the top of the triangle below and moving to adjacent numbers on the row below, the maximum total from top to bottom is 23.

       3
      7 4
     2 4 6
    8 5 9 3

    That is, 3 + 7 + 4 + 9 = 23.

    Find the maximum total from top to bottom of the triangle below:

                  75
                 95 64
                17 47 82
               18 35 87 10
              20 04 82 47 65
             19 01 23 75 03 34
            88 02 77 73 07 63 67
           99 65 04 28 06 16 70 92
          41 41 26 56 83 40 80 70 33
         41 48 72 33 47 32 37 16 94 29
        53 71 44 65 25 43 91 52 97 51 14
       70 11 33 28 77 73 17 78 39 68 17 57
      91 71 52 38 17 14 91 43 58 50 27 29 48
     63 66 04 68 89 53 67 30 73 16 69 87 40 31
    04 62 98 27 23 09 70 98 73 93 38 53 60 04 23

    NOTE: As there are only 16384 routes, it is possible to solve this problem by trying every route. However, Problem 67, is the same 
    challenge with a triangle containing one-hundred rows; it cannot be solved by brute force, and requires a clever method! ;o)
    """
    triangle = []
    for line in open(filename, "r"):
        triangle.append([int(i) for i in line.split()])

    return __triangle_max_sum(triangle)

def problem_19():
    """
    You are given the following information, but you may prefer to do some research for yourself.

    1 Jan 1900 was a Monday.

    Thirty days has September,
    April, June and November.
    All the rest have thirty-one,
    Saving February alone,
    Which has twenty-eight, rain or shine.
    And on leap years, twenty-nine.

    A leap year occurs on any year evenly divisible by 4, but not on a century unless it is
    divisible by 400.

    How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 
    to 31 Dec 2000)?
    """
    sundays = 0
    day = 0
    for year in range(1900, 1901):
        for month in range(1, 13):
            for date in range(1, __days_in_month(month, year) + 1):
                day = (day + 1) % 7

    for year in range(1901, 2001):
        for month in range(1, 13):
            for date in range(1, __days_in_month(month, year) + 1):
                if day == 6 and date == 1:
                    sundays += 1
                day = (day + 1) % 7
    return sundays

def problem_20(): 
    """
    n! means n  (n  1)  ...  3  2  1

    For example, 10! = 10  9  ...  3  2  1 = 3628800,
    and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.

    Find the sum of the digits in the number 100!
    """
    return sum(int(i) for i in str(factorial(100)))

def problem_21():
    """
    Let d(n) be defined as the sum of proper divisors of n (numbers less than n which divide 
    evenly into n).
    If d(a) = b and d(b) = a, where a != b, then a and b are an amicable pair and each of a and b 
    are called amicable numbers.

    For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; 
    therefore d(220) = 284. The proper divisors of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.

    Evaluate the sum of all the amicable numbers under 10000.
    """
    amicable = set()
    prime_list = eratosthenes(10000)

    for i in range(2, 10000):
        d_i = proper_divisor_sum(i, prime_list)
        if i != d_i and i == proper_divisor_sum(d_i):
            amicable.add(i)
            amicable.add(d_i)

    return sum(amicable)

def problem_22(filename="problem_22.dat"):
    """
    Using names.txt (right click and 'Save Link/Target As...'), a 46K text file containing
    over five-thousand first names, begin by sorting it into alphabetical order. Then 
    working out the alphabetical value for each name, multiply this value by its alphabetical
    position in the list to obtain a name score.

    For example, when the list is sorted into alphabetical order, COLIN, which is
    worth 3 + 15 + 12 + 9 + 14 = 53, is the 938th name in the list. So, COLIN would obtain 
    a score of 938  53 = 49714.

    What is the total of all the name scores in the file?
    """
    with open(filename, "r") as f:
        names = f.readline().replace("\"", "").split(",")
        names.sort()
        alph_scores = sum((i + 1) * __alphabetical_value(names[i]) for i in range(len(names)))
        return alph_scores

def problem_23():
    """
    A perfect number is a number for which the sum of its proper divisors is exactly equal to 
    the number. For example, the sum of the proper divisors of 28 would be 1 + 2 + 4 + 7 + 14 = 28,
    which means that 28 is a perfect number.

    A number n is called deficient if the sum of its proper divisors is less than n and it is 
    called abundant if this sum exceeds n.

    As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest number that can 
    be written as the sum of two abundant numbers is 24. By mathematical analysis, it can be 
    shown that all integers greater than 28123 can be written as the sum of two abundant numbers. 
    However, this upper limit cannot be reduced any further by analysis even though it is known 
    that the greatest number that cannot be expressed as the sum of two abundant numbers is 
    less than this limit.

    Find the sum of all the positive integers which cannot be written as the sum of two abundant
    numbers.
    """
    abundant = set([i for i in __abundant_below(28124)])
    return sum(i for i in range(1, 28124) if not __as_sum(abundant, i))

def problem_24():
    """
    A permutation is an ordered arrangement of objects. For example, 3124 is one possible 
    permutation of the digits 1, 2, 3 and 4. If all of the permutations are listed numerically or alphabetically, we call it lexicographic order. The lexicographic permutations of 0, 1 and 2 are:
    012   021   102   120   201   210

    What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?
    """
    return __lex_permutations({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})[999999]

def problem_25():
    """
    The Fibonacci sequence is defined by the recurrence relation:

    F[n] = F[n−1] + F[n−2], where F[1] = 1 and F[2] = 1.

    Hence the first 12 terms will be:

    F[1] = 1
    F[2] = 1
    F[3] = 2
    F[4] = 3
    F[5] = 5
    F[6] = 8
    F[7] = 13
    F[8] = 21
    F[9] = 34
    F[10] = 55
    F[11] = 89
    F[12] = 144

    The 12th term, F[12], is the first term to contain three digits.

    What is the first term in the Fibonacci sequence to contain 1000 digits?
    """
    # definition differences, Fibo starts as 1, 1, 2 ... in some places, as
    # 1, 2, 3, 5 ... in others and as 0, 1, 1, 2 ... in elsewhere. This task needs
    # the 1, 1 thing, my implementation is 1, 2 [..], hence the +1. The other +1 is
    # because Python indexes from 0.
    for i, elem in enumerate(fibonacci()):
        if len(str(elem)) >= 1000:
            return i + 1 + 1

def problem_26():
    """
    A unit fraction contains 1 in the numerator. The decimal representation of the unit fractions with
    denominators 2 to 10 are given:

    ^1/[2]  =  0.5        
    ^1/[3]  =  0.(3)      
    ^1/[4]  =  0.25       
    ^1/[5]  =  0.2        
    ^1/[6]  =  0.1(6)     
    ^1/[7]  =  0.(142857) 
    ^1/[8]  =  0.125      
    ^1/[9]  =  0.(1)      
    ^1/[10] =  0.1        
    
    Where 0.1(6) means 0.166666..., and has a 1-digit recurring cycle. It can be seen that ^1/[7] has a
    6-digit recurring cycle.

    Find the value of d < 1000 for which ^1/[d] contains the longest recurring cycle in its decimal
    fraction part.    
    """
    # Every rational number is either a recurrent or terminating decimal. The ones with a denominator
    # consisting of only 2 and 5 are terminating. Everything else is recurrent.
    rl = [__reciprocal_length(i) for i in range(1, 1000)]
    m = max(rl)
    return rl.index(m) + 1

def problem_27():
    """
    Euler published the remarkable quadratic formula:

    n² + n + 41

    It turns out that the formula will produce 40 primes for the consecutive values n = 0 to 39.
    However, when n = 40, 40^2 + 40 + 41 = 40(40 + 1) + 41 is divisible by 41, and certainly 
    when n = 41, 41² + 41 + 41 is clearly divisible by 41.

    Using computers, the incredible formula  n² − 79n + 1601 was discovered, which produces 80 primes
    for the consecutive values n = 0 to 79. The product of the coefficients, −79 and 1601, is −126479.

    Considering quadratics of the form:

    n² + an + b, where |a| < 1000 and |b| < 1000

    where |n| is the modulus/absolute value of n
    e.g. |11| = 11 and |−4| = 4

    Find the product of the coefficients, a and b, for the quadratic expression that produces the
    maximum number of primes for consecutive values of n, starting with n = 0.
    """
    max_prod = 0
    max_len = 0
    max_a = 0
    max_b = 0

    # TODO: this is just a wild guess to be honest, should that it doesn't go out of bounds somehow
    prime_list = eratosthenes(10**7)
    for a in range(-999, 1000):
        # this can't be negative, because if we start from n = 0, this is the only thing
        # keeping it above zero
        for b in range(1000):
            i = 0
            while __is_prime(i**2 + i * a + b, prime_list):
                i += 1
            if i > max_len:
                max_prod = a * b
                max_len = i
                max_a = a
                max_b = b
    return max_prod

def problem_28():
    """
    Starting with the number 1 and moving to the right in a clockwise direction a 5 by 5 spiral is
    formed as follows:

    21 22 23 24 25
    20  7  8  9 10
    19  6  1  2 11
    18  5  4  3 12
    17 16 15 14 13

    It can be verified that the sum of the numbers on the diagonals is 101.

    What is the sum of the numbers on the diagonals in a 1001 by 1001 spiral formed in the same way?
    """
    # lower-right diagonal formula: prev_edge² + prev_edge + 1
    # 1² + 1 + 1 = 3
    # 3² + 3 + 1 = 13
    # 5² + 5 + 1 = 31

    # top-left diagonal formula: curr_edge² - (curr_edge - 1)
    # 1² - (1 - 1) = 1
    # 3² - (3 - 1) = 7
    # 5² - (5 - 1) = 21

    # the other two are (even)²+1 and (odd)²

    top_left = 1
    bottom_right = 1
    # init to -1 because in the first iteration both diagonals get added
    res = -1
    for _ in range(1001):
        if top_left == bottom_right:
            res += top_left**2 - (top_left - 1)
            res += top_left**2
            top_left += 2
        else:
            res += bottom_right**2 + bottom_right + 1
            res += (bottom_right + 1)**2 + 1
            bottom_right += 2
    return res

def problem_29():
    """
    Consider all integer combinations of a^b for 2 ≤ a ≤ 5 and 2 ≤ b ≤ 5:

    2^2=4, 2^3=8, 2^4=16, 2^5=32
    3^2=9, 3^3=27, 3^4=81, 3^5=243
    4^2=16, 4^3=64, 4^4=256, 4^5=1024
    5^2=25, 5^3=125, 5^4=625, 5^5=3125

    If they are then placed in numerical order, with any repeats removed, we get the following 
    sequence of 15 distinct terms:

    4, 8, 9, 16, 25, 27, 32, 64, 81, 125, 243, 256, 625, 1024, 3125

    How many distinct terms are in the sequence generated by a^b for 2 ≤ a ≤ 100 and 2 ≤ b ≤ 100?
    """
    return len({a**b for a in range(2, 101) for b in range(2, 101)})

def problem_30():
    """
    Surprisingly there are only three numbers that can be written as the sum of fourth powers of their digits:

    1634 = 1^4 + 6^4 + 3^4 + 4^4
    8208 = 8^4 + 2^4 + 0^4 + 8^4
    9474 = 9^4 + 4^4 + 7^4 + 4^4

    As 1 = 1^4 is not a sum it is not included.

    The sum of these numbers is 1634 + 8208 + 9474 = 19316.

    Find the sum of all the numbers that can be written as the sum of fifth powers of their digits.
    """
    # heuristic for lookup, no use recalcing pows each time
    pows = [i**5 for i in range(10)]

    # Because the number we're getting is a sum, the max sum we can get is 9^5 + 9^5 + 9^5 ... = n * 9^5 = 59049n
    # 10 = 10^1, 100 = 10^2. Obviously, 10^n grows faster than 9^5n. We can get the equality point (and therefore, the
    # sum's upper bound) by solving 10^n = 59049n for n.
    # n ~ 0.0000169357; n ~ 5.51257
    # Therefore, ONLY numbers smaller than 10^5.51257 can be written as the sum of their digits. The precise number is
    # k * 9^5, but we can get k as the number of digits in 10^5.51257, which is obviously 6.

    s = 0
    for i in range(2, 6 * pows[9]):
        if i == sum(pows[int(j)] for j in str(i)):
            s += i
    return s

def problem_31():
    """
    In England the currency is made up of pound, £, and pence, p, and there are eight coins in general circulation:
    1p, 2p, 5p, 10p, 20p, 50p, £1 (100p) and £2 (200p).

    It is possible to make £2 in the following way:
    1*£1 + 1*50p + 2*20p + 1*5p + 1*2p + 3*1p

    How many different ways can £2 be made using any number of coins?
    """
    return len(__coin_permutations([1, 2, 5, 10, 20, 50, 100, 200], 200))

def problem_32():
    """
    We shall say that an n-digit number is pandigital if it makes use of all the digits 1 to n exactly once; for example, the 5-digit 
    number, 15234, is 1 through 5 pandigital.

    The product 7254 is unusual, as the identity, 39 * 186 = 7254, containing multiplicand, multiplier, and product is 1 through 9 pandigital.

    Find the sum of all products whose multiplicand/multiplier/product identity can be written as a 1 through 9 pandigital.

    HINT: Some products can be obtained in more than one way so be sure to only include it once in your sum.
    """
    # There's 9! = 362880 permutations of [1..9], if there's a linear (or near-linear) way to check whether a permutation
    # can be the identity, that should suffice for a brute solution.
    # TODO: obviously only certain digit-ed numbers need to be checked, e.g., a 1-digit by 1-digit product is max 2-digits, 
    prod_set = set()
    perms = __lex_permutations({i for i in range(1, 10)})
    for perm in perms:
        mult, f1, f2, p = __is_mult(perm)
        if mult:
            prod_set.add(p)

    return sum(prod_set)

def problem_33():
    """
    The fraction 49/98 is a curious fraction, as an inexperienced mathematician in attempting to simplify it may incorrectly believe 
    that 49/98 = 4/8, which is correct, is obtained by cancelling the 9s.

    We shall consider fractions like, 30/50 = 3/5, to be trivial examples.

    There are exactly four non-trivial examples of this type of fraction, less than one in value, and containing two digits in the 
    numerator and denominator.

    If the product of these four fractions is given in its lowest common terms, find the value of the denominator.
    """
    numerator, denominator = 1, 1
    for i in range(10, 99):
        for j in range(i + 1, 100):
            common_digits = set(str(i)).intersection(set(str(j)))
            gcd_ij = fractions.gcd(i, j)
            if gcd_ij > 1 and len(common_digits) and i % 10 != 0 and j % 10 != 0:
                # although there can only be 0 or 1 digits in common, this is easier for now
                digit = [_ for _ in common_digits][0]

                new_numerator = int(str(i).replace(digit, "", 1))
                new_denominator = int(str(j).replace(digit, "", 1))
                gcd_new = fractions.gcd(new_numerator, new_denominator)
                
                if new_numerator // gcd_new == i // gcd_ij and new_denominator // gcd_new == j // gcd_ij:
                    numerator *= new_numerator
                    denominator *= new_denominator

    return denominator // fractions.gcd(numerator, denominator)

def problem_34():
    """
    145 is a curious number, as 1! + 4! + 5! = 1 + 24 + 120 = 145.

    Find the sum of all numbers which are equal to the sum of the factorial of their digits.

    Note: as 1! = 1 and 2! = 2 are not sums they are not included.
    """
    factorials = [factorial(i) for i in range(10)]

    # Very similar to problem 30, there's an upper bound to calculate to. The max possible sum is 9! + 9! ... = n * 9!
    # This is upper bounded by 10^n somewhere (because 1 * 9! > 10^1, 2 * 9! > 10^2, etc., but 10^n is exponential,
    # while n9! is linear.
    # Solve 10^n = 362880n for n
    # n ~ 2.75575e-6; n ~ 6.36346

    s = 0
    for i in range(10, 7 * factorials[9]):
        if i == sum(factorials[int(j)] for j in str(i)):
            s += i
    return s

def problem_35():
    """
    The number, 197, is called a circular prime because all rotations of the digits: 197, 971, and 719,
    are themselves prime.

    There are thirteen such primes below 100: 2, 3, 5, 7, 11, 13, 17, 31, 37, 71, 73, 79, and 97.

    How many circular primes are there below one million?
    """
    primes = eratosthenes(10**6)
    num = 0
    for p in primes:
        all_primes = True
        for i in __rotations(p):
            if not __is_prime(i, primes):
                all_primes = False
                break
        num += all_primes

    return num

def problem_36():
    """
    The decimal number, 585 = 1001001001[2] (binary), is palindromic in both bases.

    Find the sum of all numbers, less than one million, which are palindromic in base 10 and base 2.

    (Please note that the palindromic number, in either base, may not include leading zeros.)
    """
    return sum(i for i in range(10**6) if is_palindrome(str(i)) and is_palindrome(bin(i)[2:]))

def problem_37():
    """
    The number 3797 has an interesting property. Being prime itself, it is possible to 
    continuously remove digits from left to right, and remain prime at each stage: 3797, 797, 
    97, and 7. Similarly we can work from right to left: 3797, 379, 37, and 3.

    Find the sum of the only eleven primes that are both truncatable from left to right and 
    right to left.

    NOTE: 2, 3, 5, and 7 are not considered to be truncatable primes.
    """
    primes = eratosthenes(10**6)
    s = 0
    found = 0
    for p in primes:
        if __is_truncatable(p, primes):
            found += 1
            s += p

    if found - 4 != 11:
        raise ValueError("Should have found 11 trucatable primes")

    return s - sum([2, 3, 5, 7])

def problem_38():
    """
    Take the number 192 and multiply it by each of 1, 2, and 3:

    192 × 1 = 192
    192 × 2 = 384
    192 × 3 = 576

    By concatenating each product we get the 1 to 9 pandigital, 192384576. We will call 
    192384576 the concatenated product of 192 and (1,2,3)

    The same can be achieved by starting with 9 and multiplying by 1, 2, 3, 4, and 5, giving 
    the pandigital, 918273645, which is the concatenated product of 9 and (1,2,3,4,5).

    What is the largest 1 to 9 pandigital 9-digit number that can be formed as the 
    concatenated product of an integer with (1,2, ... , n) where n > 1?
    """

    # A 9-digit number can be formed in several ways. We can observe, that it is impossible
    # for n to be larger than 9. The smallest possible 9-digit pandigital we can obtain is
    # 1 * (1, 2, .. , 9) = 123456789
    # As we increase the first factor, n decreases, because, e.g., multiplying a 2 digit number
    # by a 1 digit number yields a 2 or 3 digit number. That way, n drops to 4 (also, at least
    # one of the products is a 3 digit number, because 1 digit numbers are impossible to
    # obtain at this point). Taking a 3 digit constant, we get 3 or 4 digit products, again,
    # n drops to 3.
    # Hereby, we have proved that the constant is upper-bounded by the first 5-digit 
    # number 10000, because that is guaranteed to produce a 5 digit product and we need 
    # a 4 digit beast to get the 9 digit pandigital.
    # Since we need to find the LARGEST pandigital number, we start from 

    # Oh, btw, it's not guaranteed that just maxing the front constant guarantees the 
    # largest pandigital, so don't just return the first match

    res = 0
    for c in range(9999, -1, -1):
        s = str()
        i = 1
        while len(s) < 9:
            s += str(c * i)
            i += 1
        
        if len(s) == 9 and len(set(s).difference({'0'})) == 9:
            res = max(int(s), res)
    return res

def problem_39():
    """
    If p is the perimeter of a right angle triangle with integral length sides, {a,b,c}, 
    there are exactly three solutions for p = 120.

    {20,48,52}, {24,45,51}, {30,40,50}

    For which value of p ≤ 1000, is the number of solutions maximised?
    """
    
    sq = [i**2 for i in range(501)]
    sq_set = set(sq).difference({0})

    perimeters = [0 for _ in range(1001)]

    # A single side cannot be > 500 if the perimeters are supposed to be below 1000
    for i in range(2, 501):
        additives = __as_sum_full(sq_set, sq[i])
        for a, b in additives:
            p = isqrt(a) + isqrt(b) + i
            if p <= 1000:
                perimeters[p] += 1

    max_idx, max_val = 0, 0
    for i, v in enumerate(perimeters):
        if v > max_val:
            max_idx = i
            max_val = v
            
    return max_idx

def problem_40():
    """
    An irrational decimal fraction is created by concatenating the positive integers:

    0.123456789101112131415161718192021...

    It can be seen that the 12^th digit of the fractional part is 1.

    If d[n] represents the n^th digit of the fractional part, find the value of the following
    expression.

    d[1] × d[10] × d[100] × d[1000] × d[10000] × d[100000] × d[1000000]
    """
    # TODO: do it the right way. This takes 10e6 bytes of memory, that can easily be avoided
    s = str()
    i = 1
    while len(s) < 1000000:
        s += str(i)
        i += 1

    return int(s[0]) * int(s[9]) * int(s[99]) * int(s[999]) * int(s[9999]) * int(s[99999]) * int(s[999999])

def problem_41():
    """
    We shall say that an n-digit number is pandigital if it makes use of all the digits 1 to n exactly
    once. For example, 2143 is a 4-digit pandigital and is also prime.

    What is the largest n-digit pandigital prime that exists?
    """
    # By simple divisibility rules we can deduce that 8 and 9 digit pandigitals are ALL not
    # prime, because sum(1..9) == 45, which divides by 9, and sum(1..8) == 36, which divides
    # by 3 and 9.
    # By similar logic, we get that the only viable options for primes are 7 and 4 digit pandigitals
    # this leaves us with 4! + 7! = 5064 permutations to check, which is ridiculous...

    primes = eratosthenes(isqrt(7654321) + 2)
    for p in reversed(__lex_permutations({i for i in range(1, 8)})):
        num = int(str().join(str(i) for i in p))
        if __is_prime(num, primes):
            return num

    for p in reversed(__lex_permutations({i for i in range(1, 5)})):
        num = int(str().join(str(i) for i in p))
        if __is_prime(num, primes):
            return num

def problem_42(filename="problem_42.dat"):
    """
    The nth term of the sequence of triangle numbers is given by, tn = ½n(n+1); so the first ten triangle numbers are:

    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...

    By converting each letter in a word to a number corresponding to its alphabetical position and adding these values 
    we form a word value. For example, the word value for SKY is 19 + 11 + 25 = 55 = t10. If the word value is a triangle
    number then we shall call the word a triangle word.

    Using words.txt (right click and 'Save Link/Target As...'), a 16K text file containing nearly two-thousand 
    common English words, how many are triangle words?
    """
    res = 0
    with open(filename, "r") as f:
        words = f.readline().replace("\"", "").split(',')
        alph_scores = [__alphabetical_value(w) for w in words]
        max_score = max(alph_scores)
        triangle_nums = set()

        i = 1
        while triangle_num(i) < max_score:
            triangle_nums.add(triangle_num(i))
            i += 1
            
        return sum(score in triangle_nums for score in alph_scores)

def problem_43():
    """
    The number, 1406357289, is a 0 to 9 pandigital number because it is made up of each of the 
    digits 0 to 9 in some order, but it also has a rather interesting sub-string divisibility
    property.

    Let d1 be the 1st digit, d2 be the 2nd digit, and so on. In this way, we note the following:

    d2d3d4 = 406 is divisible by 2
    d3d4d5 = 063 is divisible by 3
    d4d5d6 = 635 is divisible by 5
    d5d6d7 = 357 is divisible by 7
    d6d7d8 = 572 is divisible by 11
    d7d8d9 = 728 is divisible by 13
    d8d9d10 = 289 is divisible by 17
    Find the sum of all 0 to 9 pandigital numbers with this property.
    """
    def __div_perms(two_digit, digit_set, divisible_by):
        """
        Generate 3-digit numbers from two_digit and an additional digit in digit_set
        that are divisible by divisible_by
        """
        for digit in digit_set:
            new_num = int(str(two_digit) + str(digit))
            if new_num % divisible_by == 0:
                yield digit

    res = 0

    # Overall, there's 10 choose 3 = 120 triplets, 3! permutations for each triplet. Since we 
    # also have to choose a digit for d1, we can do that in 9 choose 1 = 9 ways.
    # Then there are the obvious (easily testable) restrictions for some triplets: 
    # d4 has to be pair (for d2d3d4 ro be divisible by 2)
    # d3 + d4 + d5 has to be divisible by 3
    # d6 has to be 0 or 5 (for d4d5d6 to be divisible by 5)
    # 
    # There are divisibility tests for 7, 11, 13 and 17, but the given ones should already reduce the 
    # possibilities to a small enough set for brute force to be viable.
    digits = {i for i in range(10)}
    triplets = __combinations(digits, 3)
    
    # candidates for d3, d4, d5
    by_3 = [c for c in triplets if sum(c) % 3 == 0]
    
    # since d4 has to be pair, we can eliminate triplets with all odd numbers
    by_3 = [s for s in by_3 if s.intersection({0, 2, 4, 6, 8})]

    # TODO: this is fugly, the loops could & should probably be refactored into one
    # separate, that just finds the i-th number of the sequence as needed

    for c in by_3:
        for p in __lex_permutations(c):
            # only if d4 is pair
            if p[1] % 2 == 0:
                # at this point, d3, d4 and d5 are fixed, but d6 can only be
                # 0 or 5, so just fix d6 to either and work both cases.
                # With 4 numbers fixed, this theoretically leaves us 
                # 6 choose 3 = 20 options, but since we have ALREADY chosen
                # d3, d4, d5 and d6, and d5d6d7 has to be divisible by 7, we just 
                # need to find our available options, then, similarly, do the same
                # for d6d7d8 to be divisible by 11, yadda yadda. In the end, if we even
                # get there, slap the remaining two elements up in the front, permute 
                # them and live happily ever after
                for d6 in {0, 5}.difference(p):
                    for d7 in __div_perms(str(p[2]) + str(d6), digits.difference(p + [d6]), 7):
                        for d8 in __div_perms(str(d6) + str(d7), digits.difference(p + [d6, d7]), 11):
                            for d9 in __div_perms(str(d7) + str(d8), digits.difference(p + [d6, d7, d8]), 13):

                                for d10 in __div_perms(str(d8) + str(d9), digits.difference(p + [d6, d7, d8, d9]), 17):
                                    for front in __lex_permutations(digits.difference(p + [d6, d7, d8, d9, d10])):
                                        # first digit can't be zero
                                        if front[0]:
                                            res += int(str().join(str(i) for i in front + p + [d6, d7, d8, d9, d10]))

    return res

def problem_44():
    """
    Pentagonal numbers are generated by the formula, Pn=n(3n1)/2. The first ten 
    pentagonal numbers are:

    1, 5, 12, 22, 35, 51, 70, 92, 117, 145, ...

    It can be seen that P4 + P7 = 22 + 70 = 92 = P8. However, their difference, 
    70 - 22 = 48, is not pentagonal.

    Find the pair of pentagonal numbers, Pj and Pk, for which their sum and difference
    is pentagonal and D = |Pk - Pj| is minimised; what is the value of D?
    """
    # TODO: the first matching pair actually works, but no idea why

    # The difference between each consecutive member of the list is
    # 3n + 1. This can be seen by performing a simple polynomial subtraction
    # (n+1)(3(n+1) - 1)/2 - n(3n - 1)/2 = 3n + 1.
    #
    # Therefore, any pentagonal number that has the form of 3n + 1 CAN be expressed
    # as a difference of two consecutive terms.

    # Every pentagonal number is a sum of all differences up to it, e.g.,
    # P_4 = 3*0+1 + 3*1+1 + 3*2+1 + 3*3+1 = 3*(0 + 1 + 2 + 3) + 4 = 3 * 6 + 4 = 22
    # P_5 = 3 * (0 + 1 + .. + 4) + 5 = 3 * 10 + 5 = 35
    # We can easily notice that (0 + .. + n) is a triangle number
    # and extract this new formula of 
    # P_n = 3 * T_{n-1} + n = 3 * T_n - 2n
    # It is proved by opening up the Pentagon formula and adding in a (+3n - 3n) part.

    pentagon_nums = set()
    i = 1
    min_diff = float("inf")
    while True:
        curr_pent = pentagon_num(i)
        additives = __as_sum_full(pentagon_nums, curr_pent)
        for j1, j2 in additives:
            # check all pairs, whether their diff is pentagonal
            if __is_pent_number(abs(j2 - j1)):
                if abs(j1 - j2) < min_diff:
                    min_diff = abs(j1 - j2)
                    return abs(j1 - j2)

        # These are the two consecutive pentagonals that sum up to the current pentagonal
        # number (or larger). This is the minimum difference we could have, so when this
        # outgrows the currently found min_diff, we know there can be no smaller D
        # However, this takes FOREVER, so screw it for now
#        large_half = pentagon_num(math.ceil(__reverse_pent_number(curr_pent / 2)))
#        small_half = pentagon_num(math.ceil(__reverse_pent_number(curr_pent / 2)) - 1)
#        if min_diff < abs(large_half - small_half):
#            return min_diff

        pentagon_nums.add(curr_pent)
        i += 1

def problem_45():
    """
    Triangle, pentagonal, and hexagonal numbers are generated by the following formulae:

    Triangle	 	T_n = n(n + 1)/2	 	1, 3, 6, 10, 15, ...
    Pentagonal	 	P_n = n(3n - 1)/2	 	1, 5, 12, 22, 35, ...
    Hexagonal	 	H_n = n(2n - 1) 	 	1, 6, 15, 28, 45, ...
    It can be verified that T_285 = P_165 = H_143 = 40755.

    Find the next triangle number that is also pentagonal and hexagonal.
    """
    # brute force
    i = 286
    while True:
        num = triangle_num(i)
        if __is_pent_number(num) and __is_hex_number(num):
            return num
        i += 1

def problem_46():
    """
    It was proposed by Christian Goldbach that every odd composite number can be written as the sum of a prime and twice a square.

    9 = 7 + 2*1²
    15 = 7 + 2*2²
    21 = 3 + 2*3²
    25 = 7 + 2*3²
    27 = 19 + 2*2²
    33 = 31 + 2*1²

    It turns out that the conjecture was false.

    What is the smallest odd composite that cannot be written as the sum of a prime and twice a square?
    """
    prime_list = eratosthenes(10**5)
    i = 3
    while i < prime_list[-1]**2:
        if not __is_prime(i, prime_list):
            found = False
            for j in range(1, math.floor(math.sqrt(i // 2)) + 1):
                if __is_prime(i - 2 * j**2, prime_list):
#                    print("{} = {} + 2 * {}²".format(i, i - 2 * j**2, j))
                    found = True
                    break
            if not found:
                return i
        i += 2

def problem_47():
    """
    The first two consecutive numbers to have two distinct prime factors are:

    14 = 2 x 7
    15 = 3 x 5

    The first three consecutive numbers to have three distinct prime factors are:

    644 = 2² x 7 x 23
    645 = 3 x 5 x 43
    646 = 2 x 17 x 19.

    Find the first four consecutive integers to have four distinct primes factors. What is the first of these numbers?
    """
    prime_list = eratosthenes(10**5)
    
    i = 2
    while i < prime_list[-1]**2:
        curr_factors = len(factorize(i, prime_list))
        if curr_factors == 4:
            above = i
            above_factors = 4
            while above_factors == 4:
                above += 1
                above_factors = len(factorize(above, prime_list))
            above -= 1
            
            below = i
            below_factors = 4
            while below_factors == 4:
                below -= 1
                below_factors = len(factorize(below, prime_list))
            below += 1

            if above - below == 3:
                return below

            i = above + 1
        else:
            i += 4

def problem_48():
    """
    The series, 1^1 + 2^2 + 3^3 + ... + 10^10 = 10405071317.

    Find the last ten digits of the series, 11 + 22 + 33 + ... + 10001000.
    """
    return str(sum(i**i for i in range(1, 1001)))[-10:]

def problem_49():
    """
    The arithmetic sequence, 1487, 4817, 8147, in which each of the terms increases by 3330, is unusual in two ways: 
    (i) each of the three terms are prime, and, 
    (ii) each of the 4-digit numbers are permutations of one another.

    There are no arithmetic sequences made up of three 1-, 2-, or 3-digit primes, exhibiting this property, but there is one 
    other 4-digit increasing sequence.

    What 12-digit number do you form by concatenating the three terms in this sequence?
    """

    primes = set(i for i in eratosthenes(10**4) if i > 1000)
    permutations = {}

    for p in primes:
        key = "".join(sorted(str(p)))
        if key not in permutations:
            permutations[key] = [p]
        else:
            permutations[key].append(p)

    permutations = {k: v for k, v in permutations.items() if len(v) >= 3}

    for perm in permutations.values():
        perm.sort()

        for idx, first in enumerate(perm[:-2]):
            for second in perm[idx+1:-1]:
                if 2 * second - first in perm and first != 1487 and second != 4817:
                    return "".join([str(first), str(second), str(2 * second - first)])

def problem_50():
    """
    The prime 41, can be written as the sum of six consecutive primes:

    41 = 2 + 3 + 5 + 7 + 11 + 13

    This is the longest sum of consecutive primes that adds to a prime below one-hundred.

    The longest sum of consecutive primes below one-thousand that adds to a prime, contains 21 terms, and
    is equal to 953.

    Which prime, below one-million, can be written as the sum of the most consecutive primes?
    """
    # The number of primes has to be odd, because all primes (but 2) are odd, and the sum of 
    # an even number of odd numbers will be even, hence, composite

    primes = eratosthenes(10**6)
    prime_set = set(primes)
    max_len = 0

    # We need to get the prime right before one-third upperbound, because after that, there can be
    # no sequence larger than 1, because:
    #     1) any three consecutive primes exceed the upper bound;
    #     3) any two consecutive primes summed form an even number
    largest_idx = len(primes) - 1
    while primes[largest_idx] > primes[-1] // 3:
        largest_idx -= 1
    largest_idx += 1

    prime_sum_lens = [0 for _ in primes]
    s = sum(primes[:largest_idx + 1])
    
    curr_sum = s
    end_idx = largest_idx
    while not curr_sum in prime_set and end_idx > 0:
        curr_sum -= primes[end_idx]
        end_idx -= 1

    max_len = prime_sum_lens[0] = end_idx + 1
    max_idx = 0

    for i in range(1, largest_idx):
        s -= primes[i - 1]
        
        # If the max_len * current_prime is larger than our upper bound, then
        # current_prime + max_len primes larger than current_prime are DEFINITELY
        # gonna be larger than our upper bound. Any prime above will also surely
        # have this property, so we can just quit right here.
        if max_len * primes[i] > primes[-1]:
            break

        curr_sum = s
        end_idx = largest_idx

        while not curr_sum in prime_set:
            curr_sum -= primes[end_idx]
            end_idx -= 1

        if end_idx - i + 1 > max_len:
            max_len = end_idx - i + 1
            max_idx = i

    return sum(primes[max_idx:max_idx+max_len])

def problem_51():
    """
    By replacing the 1st digit of *3, it turns out that six of the nine possible values: 13, 23, 43, 53,
    73, and 83, are all prime.

    By replacing the 3rd and 4th digits of 56**3 with the same digit, this 5-digit number is the first
    example having seven primes among the ten generated numbers, yielding the family: 56003, 56113,
    56333, 56443, 56663, 56773, and 56993. Consequently 56003, being the first member of this family, is
    the smallest prime with this property.

    Find the smallest prime which, by replacing part of the number (not necessarily adjacent digits) with
    the same digit, is part of an eight prime value family.
    """
    pass
