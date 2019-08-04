Title: Math Symbols Explained with Python
Date: 2019-08-03 06:53
Modified: 2019-08-03 06:53
Category: math
Slug: math-for-programmers
Summary: Learn the meaning behind mathematical symbols used in Machine Learning using your knowledge of Python.
Status: published


When working with Machine Learning projects, you will come across a wide variety of equations that you need to implement in code. Mathematical notations capture a concept so eloquently but unfamiliarity with them makes them obscure.

In this post, I'll be explaining the most common math notations by connecting it with its analogous concept in Python. Once you learn them, you will be able to intuitively grasp the intention of an equation and be able to implement it in code.
<pre class="math">
\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y_i})^2
</pre>



## Indexing
<pre class="math">
x_i
</pre>

This symbol is taking the value at i<tt class="math">^{th}</tt> index of a vector.
```python
x = [10, 20, 30]
i = 0
print(x[i]) # 10
``` 

This can be extended for 2D vectors and so on.
<pre class="math">
x_{ij}
</pre>

```python
x = [ [10, 20, 30], [40, 50, 60] ]
i = 0
j = 1
print(x[i][j]) # 20
``` 

## Sigma


<pre class="math">
\sum_{i=0}^{N} x_i
</pre>

This symbol finds the sum of all elements in a vector for a given range. In Python, it is equivalent to looping over a vector from index 0 to index n. Notice how we're using the previously explained <tt class="math">x_i</tt> symbol to get the value at index.

```python
x = [1, 2, 3, 4, 5]
result = 0
N = len(x)
for i in range(N):
 result = result + x[i]
print(result)
```
The above code can even be shortened using built-in functions in Python as
```python
x = [1, 2, 3, 4, 5]
result = sum(x)
```

## Average

<pre class="math">
\frac{1}{N}\sum_{i=0}^{N} x_i
</pre>

Here we reuse the sigma notation and divide by the number of elements to get an average.

```python
x = [1, 2, 3, 4, 5]
result = 0
N = len(x)
for i in range(n):
 result = result + x[i]
average = result / N
print(average)
```
The above code can even be shortened in Python as
```python
x = [1, 2, 3, 4, 5]
result = sum(x) / len(x)
```

## PI


<pre class="math">
\prod_{i=0}^{N} x_i
</pre>

This symbol finds the product of all elements in a vector for a given range. In Python, it is equivalent to looping over a vector from index 0 to index n and multiplying them.

```python
x = [1, 2, 3, 4, 5]
result = 1
N = len(x)
for i in range(N):
 result = result * x[i]
print(result)
```

## Pipe
The pipe symbol can mean different things based on where it's applied.

### Absolute Value
<pre class="math">
| x | 
</pre>
<pre class="math">
| y | 
</pre>

This symbol denotes the absolute value of a number i.e. without a sign.

```python
x = 10
y = -20
abs(x) # 10
abs(y) # 20
``` 
<br>

### Norm of vector
<pre class="math">
| x |
</pre>
<pre class="math">
|| x || 
</pre>

The norm is used to calculate the magnitude of a vector. In Python, this means squaring each element of an array, summing them and then taking the square root.

```python
x = [1, 2, 3]
math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
```

## Epsilon
<pre class="math">
3\ \epsilon\ X
</pre>

This symbol checks if an element is part of a set. In Python, this would be equivalent to
```python
X = {1, 2, 3}
3 in X
```

## Function
<pre class="math">
f: X \rightarrow Y
</pre>

This denotes a function which takes a domain X and maps it to range Y. In Python, it's equivalent to taking a pool of values X, doing some operation on it to calculate pool of values Y.
```python
def f(X):
 Y = ...
 return Y
```

You will encounter the following symbols in place of X and Y. Here are what they mean:
<pre class="math">
f: R \rightarrow R
</pre>

`R` means input and outputs are real numbers and can take any value (integer, float, irrational, rational).
In Python, this is equivalent to any value except complex numbers.
```python
x = 1
y = 2.5
z = math.pi
```

You will also enounter symbols such as 
<pre class="math">
f: R^d \rightarrow R
</pre>

<tt class="math">R^d</tt> means d-dimensional vector of real numbers.

Let's assume d = 2. In Python, an example can be a function that takes 2-D array and returns it's sum. It will be mapping a <tt class="math">R^d</tt> to <tt class="math">R</tt>
```python
X = [[1, 2], [3, 4]]
f = np.sum
Y = f(X)
```

## Tensors  
  
### Transpose
<pre class="math">
X^T
</pre>

This is basically exchanging the rows and columns.
In Python, this would be equivalent to
```python
X = [[1, 2, 3], 
    [4, 5, 6]]
np.transpose(X)
```  
Output would be a list with exchanged rows and columns.
```
[[1, 4], 
 [2, 5],
 [3, 6]]
```

<br>
### Element wise multiplication
<pre class="math">
z = x \odot y
</pre>

It means multiplying the corresponding elements in two tensors. In Python, this would be equivalent to multiplying the corresponding elements in two lists.
```python
x = [[1, 2], 
    [3, 4]]
y = [[2, 2], 
    [2, 2]]
z = np.multiply(x, y)
```
Output is
```
[[2, 4]],
[[6, 8]]
```
<br>
### Dot Product
<pre class="math">
XY \newline
X.Y
</pre>

It gives the sum of the products of the corresponding entries of the two sequences of numbers.
```python
X = [1, 2, 3]
Y = [4, 5, 7]
# 1*4 + 2*5 + 3*7
```
<br>
### Hat
<pre class="math">
\hat{x}
</pre>

The hat gives the unit vector. This means dividing each component in a vector by it's length(norm).
```python
x = [1, 2, 3]
length = math.sqrt(sum([e**2 for e in x]))
x_hat = [e/length for e in x]
```
This makes the magnitude of the vector 1 and only keeps the direction.
```python
math.sqrt(sum([e**2 for e in x_hat]))
# 1.0
```

## Exclamation
<pre class="math">
x!
</pre>

This denotes the factorial of a number. It is the product of numbers starting from 1 to that number. In Python, it can be calculated as
```python
x = 5
fact = 1
for i in range(x, 0, -1):
    fact = fact * i
print(fact)
```

The same thing can also be calculated using built-in function.
```python
import math
math.factorial(x)
```
The output is
```
# 5*4*3*2*1
120
```
