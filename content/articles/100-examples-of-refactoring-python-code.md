Title: Recipes for Refactoring Python Code
Date: 2019-3-30 22:8
Modified: 2019-3-30 22:8
Category: python
Tags:
Slug: refactoring-python-code
Summary: TODO
Status: draft

TODO

##1. Test for Existence
Consider a usecase where we need to print first element of a list. We check if list has elements
and then print the first element.
```python
a = [1, 2, 3]
if len(a) > 0:
    print(a[0])
```

In Python, empty data structures are treated as False as illustrated below.
```python
>>> bool([])
False
>>> bool([1, 2, 3])
True
```

So, we can exploit this feature to make the previous code succint without checking the length.
```python
a = [1, 2, 3]
if a:
    print(a[0])
```

##2. Check for range
You need to check if an age is between 18 and 65 years old.
```python
if age >=18 and age <=65:
    print("Do Something")
```

Python allows combining comparision operators to check if value falls in a range similar to a math equation. So, we can refactor the code as:
```python
if 18 <= age <=65:
    print("Do Something")
```  


##3. Multiple Equality Check
We need to check if a character is a vowel or consonant.
```python
c = 'u'
if c == 'a' or c == 'e' or c == 'i' or c =='o' or c == 'u':
    print('vowel')
else:
    print('consonant')
```

We can use the `in` operator to simplify the boolean expression.
```python
c = 'u'
if c in ('a', 'e', 'i', 'o', 'u'):
    print('vowel')
else:
    print('consonant')
```

##4. Setting default value
In a webapp, we are displaying 10 items to user by default. If we get count as attribute in request, we display that count of items.

```python
count = get_count()
count = count if count else 10
```

Here, instead of if / else, we can use the `or` operator to simplify the expression.
```python
count = get_count() or 10
```

##5: Redundant Lambda
We need to convert a list of numbers from string to integer. We use a lambda function to take each element and use the inbuilt int(...) function to convert that to integer.

```python
a = ['1', '2', '3']
nums = list(map(lambda x: int(x), a))
```

We can use the inbuilt function directly without requiring lambda function. We know `map` expects a function as the first argument that takes one parameter and since `int` is a function that takes one parameter, it can be used.
```python
a = ['1', '2', '3']
nums = list(map(int, a))
```

The same pattern can be used for functions such as sorted.
```python
sorted(a, key=int)
```

##6. Intermediate List
We need a sum of squares of numbers from 1 to 100.
```python
a = sum([i*i for i in range(100)])
```

Instead of creating list using list comprehension, we can use a generator expression for optimization. Notice the removal of square brackets. The list is not created in memory.
```python
a = sum(i*i for i in range(100))
```

##7. Boolean Result
We need to check if a person is eligible for voting or not.
```python
def is_voter(age):
    if age >= 18:
        return True
    else:
        return False
```

Here since we are returning the result of the boolean expression in if statement, we can directly return the result itself. 

```python
def is_voter(age):
    return age >= 18
```