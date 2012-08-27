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
    """
    Returns n!
    """
    return prod(i for i in range(1, n + 1))

def __num_grid_paths(x, y):
    """
    Return the number of paths without backtracking from the top-left corner to the
    bottom-right corner of an x by y grid.
    """
    # TODO: somewhat suboptimal, we don't need the full factorials
    return __factorial(x + y) // (__factorial(y) * __factorial(x))

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

def divisors(n):
    """
    Returns a set of proper divisors for n (including n).
    """
    factors = factorize(n)
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

def divisors_blunt(num):
    return [i for i in range(1, num + 1) if num % i == 0]

def divisors_of_triangle_blunt(num):

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
    return sum([len(__num_to_text(i).replace(" ", "").replace("-", "")) for i in range(1, 1001)])

def problem_18(filename):
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

    A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.

    How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?
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
    return sum(int(i) for i in str(__factorial(100)))

def problem_21():
    """
    Let d(n) be defined as the sum of proper divisors of n (numbers less than n which divide evenly into n).
    If d(a) = b and d(b) = a, where a != b, then a and b are an amicable pair and each of a and b are called amicable numbers.

    For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore d(220) = 284. The proper divisors 
    of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.

    Evaluate the sum of all the amicable numbers under 10000.
    """
    amicable = set()

    for i in range(2, 10000):
        print("Working on {}".format(i))
        d_i = sum(divisors(i).difference({i}))
        if i != d_i and i == sum(divisors(d_i).difference({d_i})):
            amicable.add(i)
            amicable.add(d_i)

    return sum(amicable)
