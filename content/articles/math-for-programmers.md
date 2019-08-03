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
print(x[i][j]) # 50
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
